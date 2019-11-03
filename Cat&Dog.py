"""
Dogs vs. Cats
Create an algorithm to distinguish dogs from cats
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.autograd import Variable
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from model_utils import *
from data_utils import *
import random
import os

use_GPU = torch.cuda.is_available()
print('Cuda', use_GPU)

num_of_pic = 0
if num_of_pic > 0:
    for i in range(num_of_pic):
        data_dir = ('D:/pro/data/Cat&Dog/train/Cat/cat.%d.jpg' %i)
        plt.figure(figsize=(16,16))
        plt.subplot(num_of_pic,1,i+1)
        img = plt.imread(data_dir)
        cv2.imshow("Pic", img)
        cv2.waitKey(500)

"""
Variable
"""
sz = 224
bs = 8
class_num = 12

"""
Load datasets
"""

# make validation dataset
Data_dir = 'D:/pro/data/Cat&Dog/'
train_dir = os.path.join(Data_dir + 'train')
valid_dir = os.path.join(Data_dir + 'valid')
test_dir = os.path.join(Data_dir + 'test')

if not os.path.exists(valid_dir):
    create_validation_data(train_dir, valid_dir, split=0.20, ext='jpg')

print("train_dir_list: ",os.listdir(train_dir))
print("valid_dir_list: ",os.listdir(valid_dir))

zoom = int((1.0 + random.random()/10.0) * sz)

train_transforms = transforms.Compose([
    transforms.Resize((zoom, zoom)),
    transforms.RandomCrop(sz),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_datasets = torchvision.datasets.ImageFolder(train_dir,
                                                   transform=train_transforms)

val_datasets = torchvision.datasets.ImageFolder(valid_dir, transform=transforms.Compose([
                                                                            transforms.Resize((sz,sz)),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

print('size of train_datasets', len(train_datasets))
print('size of valid_datasets', len(val_datasets))



train_DL = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=bs, shuffle=True)
val_DL = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=bs, shuffle=True)


"""
Model
"""
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,16,3,1,2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2)
        )
        self.layer2 = nn.Sequential(nn.Conv2d(16,32,3,1,2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2)
        )
        self.layer3 = nn.Sequential(nn.Conv2d(32,64,3,1,2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2)
        )
        self.fc = nn.Linear(113*113*16, class_num)

    def forward(self, x):
        out1 = self.layer1(x)
        #out2 = self.layer2(out1)
        #out3 = self.layer3(out2)
        out1 = out1.reshape(out1.size(0), -1)
        y = self.fc(out1)
        return y


#model = SimpleCNN()

model = load_pretrained_resnet18(model_path=None, num_classes=12)
# C:\Users\Kian/.cache\torch\checkpoints\resnet50-19c8e357.pth
if use_GPU:
    model = model.cuda()

"""
Loss
"""
loss_t = nn.CrossEntropyLoss()

"""
optim
"""
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

def to_var(x, volatile=False):
    if use_GPU:
        x = x.cuda()
    return Variable(x, volatile=volatile)


"""
Main Loop
"""
num_epochs = 5
losses = []
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, targets) in enumerate(train_DL):

        inputs = to_var(inputs)
        targets = to_var(targets)

        # forward
        optimizer.zero_grad()
        outputs = model(inputs)

        # Loss
        loss = loss_t(outputs, targets)
        losses +=[loss.data.item()]

        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()

        if (i+1) % 200==0:
            print('Train, Epoch [%2d/%2d], Step [%3d/%3d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_DL), loss.data.item()))


plt.figure(figsize=(12, 4))
plt.plot(losses)
plt.show()


def evaluate_model(model, dataloader, type):
    model.eval()  # for batch normalization layers
    corrects = 0
    for inputs, targets in dataloader:
        inputs, targets = to_var(inputs, True), to_var(targets, True)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        corrects += (preds == targets.data).sum()

    print('accuracy of {}: {:.2f}'.format(type, 100. * corrects / len(dataloader.dataset)))


evaluate_model(model, train_DL, 'Train')

evaluate_model(model, val_DL, 'Val')
