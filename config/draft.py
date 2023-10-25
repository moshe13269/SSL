import torch
from copy import deepcopy
from ema_pytorch import EMA

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from ignite.handlers.ema_handler import EMAHandler
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from transformers import Data2VecVisionConfig, Data2VecVisionModel

# Initializing a Data2VecVision data2vec_vision-base-patch16-224-in22k style configuration
configuration = Data2VecVisionConfig()
configuration.num_hidden_layers = 6 # default 12
configuration.num_attention_heads = 6 # default 12
configuration.intermediate_size = 2048 # default 3072
configuration.image_size = 32 #640
configuration.num_channels = 3 #1


# Initializing a model (with random weights) from the data2vec_vision-base-patch16-224-in22k style configuration
model = Data2VecVisionModel(configuration)
teacher = Data2VecVisionModel(configuration)
teacher.cuda()
model.cuda()

# Accessing the model configuration
configuration = model.config


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True,
                                          # num_workers=1
                                          )

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False,
                                         # num_workers=1
                                         )


criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)


ema = EMA(
    model,
    beta = 0.9999,              # exponential moving average factor
    update_after_step = 100,    # only after this number of .update() calls will it start updating
    update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
)


for epoch in range(2):

    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(images.cuda())

        teacher = deepcopy(model)

        # teacher_encoding = tf.stop_gradient(teacher_encoding * self.tau + (1 - self.tau) * student_encoding)

        trg = torch.zeros_like(outputs.last_hidden_state).requires_grad_(True).cuda()
        loss = criterion(outputs.last_hidden_state, trg.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

