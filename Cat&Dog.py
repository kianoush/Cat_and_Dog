"""
Dogs vs. Cats
Create an algorithm to distinguish dogs from cats
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
import cv2
import matplotlib.pyplot as plt

use_GPU = torch.cuda.is_available()

"""
Variable
"""
sz = 224
bs = 32
class_num = 2

"""
Load datasets
"""
train_datasets = torchvision.datasets.ImageFolder('D:/pro/data/Cat&Dog/train', transform=transforms.Compose([transforms.ToTensor(),
                                                                                                        transforms.Resize((sz,sz)),
                                                                                                        transforms.RandomHorizontalFlip(),
                                                                                                        transforms.ColorJitter(0.1,0.1,0.1,0.1),
                                                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

val_datasets = torchvision.datasets.ImageFolder('D:/pro/data/Cat&Dog/val', transform=transforms.Compose([transforms.ToTensor(),
                                                                                                        transforms.Resize((sz,sz)),
                                                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

test_datasets = torchvision.datasets.ImageFolder('D:/pro/data/Cat&Dog/val', transform=transforms.Compose([transforms.ToTensor(),
                                                                                                        transforms.Resize((sz,sz)),
                                                                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

num_of_pic = 10
for i in range(num_of_pic):
    data_dir = ('D:/pro/data/Cat&Dog/train/Cat/cat.%d.jpg' %i)
    plt.figure(figsize=(16,16))
    plt.subplot(num_of_pic,1,i+1)
    img = plt.imread(data_dir)
    cv2.imshow("Pic", img)
    cv2.waitKey(500)

train_DL = torch.utils.data.DataLoader(train_datasets, batch_size=bs, shuffle=True)
val_DL = torch.utils.data.DataLoader(val_datasets, batch_size=bs, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_datasets, batch_size=bs, shuffle=True)

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
        self.fc = nn.Linear(57*57*32, class_num)
    def forward(self,x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2 = out2.reshape(out2.size(0), -1)
        y = self.fc(out2)
        return y


model = SimpleCNN

"""
Loss
"""
loss = nn.CrossEntropyLoss()

"""
optim
"""
optimaizer = torch.optim.SGD(SimpleCNN.parameters(), lr=0.001, momentum=0.9)

if use_GPU:
    model = SimpleCNN.cuda

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)



