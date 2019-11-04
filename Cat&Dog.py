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
import sys
import time


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
class_num = 2

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
        self.fc = nn.Linear(29*29*64, class_num)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out3 = out3.reshape(out3.size(0), -1)
        y = self.fc(out3)
        return y


"""
define train
"""
def to_var(x, volatile=False):
    if use_GPU:
        x = x.cuda()
    return Variable(x, volatile=volatile)


def train_one_epoch(model, dataloder, criterion, optimizer, scheduler):
    if scheduler is not None:
        scheduler.step()

    model.train(True)

    steps = len(dataloder.dataset) // dataloder.batch_size

    running_loss = 0.0
    running_corrects = 0
    losses_trn = []

    for i, (inputs, labels) in enumerate(dataloder):
        inputs, labels = to_var(inputs), to_var(labels)

        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        losses_trn += [loss.data.item()]

        # backward
        loss.backward()

        # update parameters
        optimizer.step()

        # statistics

        running_loss = (running_loss * i + loss.data.item()) / (i + 1)

        running_corrects += torch.sum(preds == labels.data)
        sys.stdout.flush()
        sys.stdout.write("\r  Step %d/%d | Loss: %.5f" % (i, steps, loss.data.item()))

    epoch_loss = running_loss
    epoch_acc = running_corrects.data.item() / len(dataloder.dataset)
    sys.stdout.flush()
    print('\r{} Loss: {:.5f} Acc: {:.5f}'.format('  Train', epoch_loss, epoch_acc))

    return model, epoch_loss


def validate_model(model, dataloder, criterion):
    model.train(False)

    steps = len(dataloder.dataset) // dataloder.batch_size

    running_loss = 0.0
    running_corrects = 0
    losses_val = []

    for i, (inputs, labels) in enumerate(dataloder):
        inputs, labels = to_var(inputs, True), to_var(labels, True)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)


        # statistics
        running_loss = (running_loss * i + loss.data.item()) / (i + 1)
        running_corrects += torch.sum(preds == labels.data)
        sys.stdout.flush()
        sys.stdout.write("\r  Step %d/%d | Loss: %.5f" % (i, steps, loss.data.item()))

    epoch_loss = running_loss
    epoch_acc = running_corrects.data.item() / len(dataloder.dataset)
    sys.stdout.flush()
    print('\r{} Loss: {:.5f} Acc: {:.5f}'.format('  Valid', epoch_loss, epoch_acc))

    return epoch_acc, epoch_loss


def train_model1(model, train_dl, valid_dl, criterion, optimizer,
                 scheduler=None, num_epochs=10):
    if not os.path.exists('models'):
        os.mkdir('models')

    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    losses_val_t = []
    losses_trn_t = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        ## train and validate
        model, losses_trn = train_one_epoch(model, train_dl, criterion, optimizer, scheduler)
        val_acc, losses_val = validate_model(model, valid_dl, criterion)

        losses_val_t += [losses_val]
        losses_trn_t += [losses_trn]

        # deep copy the model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, "./models/epoch-{}-acc-{:.5f}.pth".format(epoch, best_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, losses_trn_t, losses_val_t


"""
Main Loop
"""

    # create model

#model = SimpleCNN()
model = load_pretrained_resnet50(model_path=None, num_classes=2)
if use_gpu:
    model = model.cuda()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

# train
model, losses_trn_t, losses_val_t = train_model1(model, train_DL, val_DL, criterion,optimizer, num_epochs=10)

plt.figure(figsize=(12, 4))
plt.plot(losses_trn_t)
plt.plot(losses_val_t)
plt.show()



