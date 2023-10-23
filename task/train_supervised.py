import os
import sys
sys.path.append('/home/moshela/work/moshe/pycharm/Projects/dataset/')
import torch
import numpy as np
import mlflow.pytorch
# from mlflow import MlflowClient
from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.utils import init_weight_model, load_model
from torch.utils.data.sampler import SubsetRandomSampler



configuration = Data2VecVisionConfig()
configuration.num_hidden_layers = 6 # default 12
configuration.num_attention_heads = 6 # default 12
configuration.intermediate_size = 2048 # default 3072
configuration.image_size = 32 #640
configuration.num_channels = 3 #1


# Initializing a model (with random weights) from the data2vec_vision-base-patch16-224-in22k style configuration
model = Data2VecVisionModel(configuration)
model.cuda()

# Accessing the model configuration
configuration = model.config


# from torch.utils.data.distributed import DistributedSampler


class TrainTaskSupervised:

    def __init__(self, cfg: DictConfig, args):

        self.cfg = cfg

        ######################################
        # random seed
        ######################################
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
        # losses & learning_rate & epochs
        ######################################
        self.num_ce_loss = self.cfg.train_task.TrainTask.get('num_ce_loss')
        if 'linear_classifier' in cfg.train_task.TrainTask.model:
            self.outputs_dimension_per_outputs = \
                cfg.train_task.TrainTask.model.linear_classifier.outputs_dimension_per_outputs
        elif 'outputs_dimension_per_outputs' in cfg.train_task.TrainTask:
            self.outputs_dimension_per_outputs = cfg.train_task.TrainTask.outputs_dimension_per_outputs
        else:
            self.outputs_dimension_per_outputs = None

        self.epochs = self.cfg.train_task.TrainTask.get('epochs')

        self.learning_rate = self.cfg.train_task.TrainTask.get('learning_rate')

        self.num_classes = self.num_ce_loss

        if self.num_ce_loss is not None:
            self.loss = losses.losses_instantiate(self.num_ce_loss,
                                                  cfg.train_task.TrainTask.loss_ce,
                                                  list(self.outputs_dimension_per_outputs),
                                                  cfg.train_task.TrainTask.loss)
        else:
            self.loss = losses.losses_instantiate(loss=cfg.train_task.TrainTask.loss)

        if not self.cfg.train_task.TrainTask.get('loss_l2'):
            self.loss = self.loss[1:]

        ################################
        # model instantiate
        ################################
        if self.path2load_model is None:

            self.model = instantiate(cfg.train_task.TrainTask.model)
            self.model.apply(init_weight_model)

            # if isinstance(self.model, model.synth_transformer_encoder.SynthTransformerEncoder) or \
            #         isinstance(self.model, model.ViT.MyViT):
            if isinstance(self.model, model.ViT.MyViT):
                pl_model = model.pl_model_ViT.LitModelEncoder(model=self.model,
                                                              losses=self.loss,
                                                              learn_rate=self.learning_rate,
                                                              outputs_dimension_per_outputs=
                                                              self.outputs_dimension_per_outputs,
                                                              num_classes=self.num_classes
                                                              )
            else:
                pl_model = model.pl_model_decoder.LitModelDecoder(model=self.model,
                                                                  losses=self.loss,
                                                                  learn_rate=self.learning_rate,
                                                                  path2save_images=self.path2images)

            # compile
            # if sys.platform != 'win32':
            #     pl_model = torch.compile(pl_model)

            print("Learning rate: %f" % self.learning_rate)
        else:
            self.model, self.optimizer = load_model(self.path2load_model)
            pl_model = self.model

        ################################
        # dataset & dataloader
        ################################
        self.num_workers = self.cfg.train_task.TrainTask.get('num_workers')
        self.batch_size = self.cfg.train_task.TrainTask.get('batch_size')

        dataset_folders = os.listdir(args.path2data)
        if 'train' in dataset_folders and 'test' in dataset_folders and 'valid' in dataset_folders:

            list2load = ['train', 'test', 'valid']
            self.dataset = {}

            for name in list2load:
                ################################
                # parser csv to labels
                ################################
                path2dataset = os.path.join(args.path2data, name)

                if not os.path.exists(os.path.join(os.getcwd(), path2dataset, 'labels')):
                    path2csv = [os.path.join(path2dataset, csv_file)
                                for csv_file in os.listdir(path2dataset) if csv_file.endswith('.csv')][0]

                    os.makedirs(os.path.join(path2dataset, 'labels'))

                    main_(path2csv=path2csv, path2save=os.path.join(path2dataset, 'labels'))

                self.dataset[name] = instantiate(cfg.train_task.TrainTask.processor)
                self.dataset[name].load_dataset(os.path.join(path2dataset, 'data'))

            train_sampler = SubsetRandomSampler(list(range(self.dataset['train'])))
            valid_sampler = SubsetRandomSampler(list(range(self.dataset['valid'])))

            train_loader = torch.utils.data.DataLoader(self.dataset['train'],
                                                       batch_size=self.batch_size['train'],
                                                       sampler=train_sampler,
                                                       num_workers=self.num_workers['train'])

            validation_loader = torch.utils.data.DataLoader(self.dataset['valid'],
                                                            batch_size=self.batch_size['valid'],
                                                            sampler=valid_sampler,
                                                            num_workers=self.num_workers['valid'])

        else:
            ################################
            # parser csv to labels
            ################################
            if not os.path.exists(os.path.join(os.getcwd(), args.path2data, 'labels')):
                path2csv = [os.path.join(args.path2data, csv_file)
                            for csv_file in os.listdir(args.path2data) if csv_file.endswith('.csv')][0]

                os.makedirs(os.path.join(args.path2data, 'labels'))

                main_(path2csv=path2csv, path2save=os.path.join(args.path2data, 'labels'))

            self.dataset = instantiate(cfg.train_task.TrainTask.processor)
            self.dataset.load_dataset(os.path.join(args.path2data, 'data'))

            dataset_size = len(self.dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(0.1 * dataset_size))
            # if shuffle_dataset:

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
                         ckpt_path=None) #r'C:\Users\moshe\PycharmProjects\checkpoints\3')

        # def print_auto_logged_info(r):
        #     tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        #     artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        #     print("run_id: {}".format(r.info.run_id))
        #     print("artifacts: {}".format(artifacts))
        #     print("params: {}".format(r.data.params))
        #     print("metrics: {}".format(r.data.metrics))
        #     print("tags: {}".format(tags))

        # mlflow.pytorch.autolog()

        # Train the model
        # with mlflow.start_run() as run:
        # print_auto_logged_info(mlflow.get_run(run_id=mlf_logger.run_id))
