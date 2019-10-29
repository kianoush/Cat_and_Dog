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
bs = 32
class_num = 2

"""
Load datasets
"""
train_datasets = torchvision.datasets.ImageFolder('D:/pro/data/Cat&Dog/train',
                                                   transform=transforms.Compose([
                                                    transforms.Resize((sz, sz)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
                                                    transforms.RandomRotation(20),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ]))

val_datasets = torchvision.datasets.ImageFolder('D:/pro/data/Cat&Dog/val', transform=transforms.Compose([
                                                                                                        transforms.Resize((sz,sz)),
                                                                                                        transforms.ToTensor(),
                                                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

test_datasets = torchvision.datasets.ImageFolder('D:/pro/data/Cat&Dog/val', transform=transforms.Compose([
                                                                                                        transforms.Resize((sz,sz)),
                                                                                                        transforms.ToTensor(),
                                                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))


train_DL = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=bs, shuffle=True)
val_DL = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=bs, shuffle=True)
test_DL = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=bs, shuffle=True)


"""
Model
"""
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,16,5,1,3),
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

model = load_pretrained_resnet50(model_path=None, num_classes=2)
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

        if (i+1) % 50==0:
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

evaluate_model(model, test_DL, 'Test')