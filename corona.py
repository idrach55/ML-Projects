import cv2
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import itertools
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def setup():
    dataset = datasets.ImageFolder('./coronahack-chest-xraydataset/dataset',transform=transforms.Compose([
                                   transforms.Resize((256,256)),
                                   np.array,
                                   torch.tensor,
                                   lambda x: torch.reshape(x,(3,256,256))/255]))
    trainset, valset = torch.utils.data.random_split(dataset, [2560, 2724])
    loaders = {'train': torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True),
               'test':  torch.utils.data.DataLoader(valset)}
    return loaders


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False)
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm2d(128)
        self.batch4 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(65536, 3)

    def forward(self, x):
        """
        Forward-propogate through the neural net.

        Apply rounds of convolution, then batch normalization, leaky-relu, and dropout.
        """
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.leaky_relu(x, negative_slope=0.20)
        x = F.dropout(x, 0.5)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.leaky_relu(x, negative_slope=0.20)
        x = F.dropout(x, 0.4)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.leaky_relu(x, negative_slope=0.20)
        x = F.dropout(x, 0.4)

        x = self.conv4(x)
        x = self.batch4(x)
        x = F.leaky_relu(x, negative_slope=0.20)
        x = F.dropout(x, 0.4)

        x = self.flatten(x)
        x = self.dense(x)
        return F.log_softmax(x, dim=1)


def get_sample(testset, N=100):
    shape   = testset.dataset[0][0].shape
    sample_data   = torch.zeros(size=(N,shape[0],shape[1],shape[2]))
    sample_target = torch.zeros(N)
    for batch_idx, (data, target) in enumerate(testset):
        if batch_idx >= N:
            break
        sample_data[batch_idx] = data
        sample_target[batch_idx] = target
    return sample_data, sample_target


def test_sample(model, testset, N=100):
    if N is None:
        N = len(testset)
    # Activate evaluation on the model.
    model.eval()
    # Vectors of actual and predicted classes.
    results = torch.zeros(N)
    targets = torch.zeros(N)

    # Iterate through N entries in the test set.
    for batch_idx, (data, target) in enumerate(testset):
        if batch_idx >= N:
            break
        # Record actual and predicted
        results[batch_idx] = model(data).argmax()
        targets[batch_idx] = target
    # Return vectors, can compute accuracy as
    # (results == targets).sum() / len(targets)
    return results, targets


def train_many(epochs, model, optimizer, trainset, testset):
    # Record loss on training set, and accuracy on test set.
    losses = []
    accuracies = []

    # Take sample from test set.
    sample_data, sample_target = get_sample(testset, N=100)
    for epoch in range(1, epochs+1):
        # Run an epoch.
        model.train()
        for batch_idx, (data, target) in enumerate(trainset):
            # Zero gradient, evaluate, backward-propogate, and step the minimization.
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            # Every-so-often print stats
            if batch_idx > 0 and batch_idx % 20 == 0:
                progress = (batch_idx+1)*64 + (epoch - 1)*len(trainset.dataset)

                print('[{:<2}/{}][{:<4}/{} ({:>2.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, epochs, (batch_idx+1)*64, len(trainset.dataset),
                    100. * progress / (epochs * len(trainset.dataset)), loss.item()))
                losses.append((progress, loss.item()))

        # Every-so-often run on test set and print stats.
        torch.save(model.state_dict(), 'corona.pt')
        #results, targets = test_sample(model, testset)
        model.eval()
        accuracy = ((model(sample_data).argmax(axis=1) == sample_target).sum() / len(sample_target)).item()
        accuracies.append((epoch*len(trainset.dataset), accuracy))
        print('[{:<2}/{}]\t Accuracy: {:.2f}%'.format(epoch, epochs, 100*accuracy))
    return model, losses, accuracies


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1).reshape(-1,1)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)