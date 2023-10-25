import os
import sys
import torch
import numpy as np
import mlflow.pytorch
from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.utils import init_weight_model, load_model
from torch.utils.data.sampler import SubsetRandomSampler


class TrainTaskSupervised:

    def __init__(self, cfg: DictConfig, args):

        self.cfg = cfg

        ######################################
        # random seed
        ######################################
        random.seed(123456)
        np.random.seed(123456)
        torch.manual_seed(123456)

        ######################################
        # model_name & path2save
        ######################################
        if args.path2save is None:
            self.path2save_model = self.cfg.train_task.TrainTask.get('path2save')
        else:
            self.path2save = args.path2save

        self.model_name = self.cfg.train_task.TrainTask.get('model_name')

        self.path2load_model = self.cfg.train_task.TrainTask.get('path2load_model')

        #######################################
        # makedir to save
        #######################################
        if not os.path.exists(os.path.join(os.getcwd(), self.path2save, 'checkpoint')):
            os.makedirs(os.path.join(self.path2save, 'checkpoint'))
        self.path2checkpoint = os.path.join(self.path2save, 'checkpoint')

        if not os.path.exists(os.path.join(os.getcwd(), self.path2save, 'images')):
            os.makedirs(os.path.join(self.path2save, 'images'))
        self.path2images = os.path.join(self.path2save, 'images')

        if not os.path.exists(os.path.join(os.getcwd(), self.path2save, 'model')):
            os.makedirs(os.path.join(self.path2save, 'model'))
        self.path2model = os.path.join(self.path2save, 'model')

        ######################################
        # losses & epochs
        ######################################
        self.epochs = self.cfg.train_task.TrainTask.get('epochs')

        self.loss = losses.losses_instantiate(loss=cfg.train_task.TrainTask.loss)

        ################################
        # model instantiate
        ################################
        configuration = Data2VecVisionConfig()
        configuration.num_hidden_layers = self.cfg.train_task.TrainTask.get('epochs')
        configuration.num_attention_heads = self.cfg.train_task.TrainTask.get('epochs')
        configuration.intermediate_size = self.cfg.train_task.TrainTask.get('epochs')
        configuration.image_size = self.cfg.train_task.TrainTask.get('epochs')
        configuration.num_channels = self.cfg.train_task.TrainTask.get('epochs')

        model = Data2VecVisionModel(configuration)

        if self.path2load_model is None:

            # self.model.apply(init_weight_model)

            if isinstance(self.model, model.ViT.MyViT):
                pl_model = model.pl_model_ViT.LitModelEncoder(model=self.model,
                                                              losses=self.loss,
                                                              learn_rate=self.learning_rate,
                                                              outputs_dimension_per_outputs=
                                                              self.outputs_dimension_per_outputs,
                                                              num_classes=self.num_classes
                                                              )

        # compile
        if sys.platform != 'win32':
            pl_model = torch.compile(pl_model)

            print("Learning rate: %f" % self.learning_rate)
        else:
            self.model, self.optimizer = load_model(self.path2load_model)
            pl_model = self.model

        ################################
        # dataset & dataloader
        ################################
        self.num_workers = self.cfg.train_task.TrainTask.get('num_workers')
        self.batch_size = self.cfg.train_task.TrainTask.get('batch_size')

        if isinstance(args.path2data, (list, tuple)):
            dataset_folders = []
            for path in args.path2data:
                dataset_folders += os.listdir(path)
        else:
            dataset_folders = os.listdir(args.path2data)

        self.dataset = instantiate(cfg.train_task.TrainTask.processor)
        self.dataset[name].load_dataset(os.path.join(path2dataset, 'data'))

        ################################
        #  split to train & valid
        ################################
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))

        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=self.batch_size['train'],
                                                   sampler=train_sampler,
                                                   num_workers=self.num_workers['train'])

        validation_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size['valid'],
                                                        sampler=valid_sampler,
                                                        num_workers=self.num_workers['valid'])

        #######################################
        # logger & checkpoint & trainer & fit
        #######################################
        mlf_logger = MLFlowLogger(experiment_name=self.model_name,
                                  tracking_uri="file:./ml-runs",
                                  save_dir=None)

        checkpoint_callback = ModelCheckpoint(dirpath=self.path2checkpoint,
                                              save_weights_only=False,
                                              monitor='validation_loss',
                                              save_last=True,
                                              verbose=True)

        if torch.cuda.is_available():
            accelerator = 'gpu'
        else:
            accelerator = 'cpu'
        print('Use the accelerator: {}'.format(accelerator))

        self.trainer = Trainer(logger=mlf_logger,
                               callbacks=[checkpoint_callback],
                               accelerator=accelerator,
                               max_epochs=self.epochs)

        self.trainer.fit(model=pl_model,
                         train_dataloaders=train_loader,
                         val_dataloaders=validation_loader,
                         ckpt_path=None)
