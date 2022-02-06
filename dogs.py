"""
Author: Isaac Drachman
Date:   11/1/2020
"""

import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

ANNOTATION_DIR = './stanford-dogs-dataset/annotations/Annotation'
IMG_DIR        = './stanford-dogs-dataset/images/Images'

# Constants for image size/channels
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNEL = 3

class Model(nn.Module):
    def __init__(self, output_dim):
        super(Model, self).__init__()

        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(IMG_CHANNEL, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.4)
        self.drop4 = nn.Dropout(0.4)
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm2d(128)
        self.batch4 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(4096, self.output_dim)
        self.dense2 = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x):
        """
        Forward-propogate through the neural net.
        Apply rounds of convolution, batch normalization, leaky-relu, and dropout.
        """
        x = self.batch1(self.conv1(x))
        x = F.leaky_relu(x, negative_slope=0.20)
        x = self.drop1(x)

        x = self.batch2(self.conv2(x))
        x = F.leaky_relu(x, negative_slope=0.20)
        x = self.drop2(x)

        x = self.batch3(self.conv3(x))
        x = F.leaky_relu(x, negative_slope=0.20)
        x = self.drop3(x)

        x = self.batch4(self.conv4(x))
        x = F.leaky_relu(x, negative_slope=0.20)
        x = self.drop4(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
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
    losses = {'train': [], 'test': []}
    accuracies = {'train': [], 'test': []}

    # Take sample from test set.
    sample_data, sample_target = get_sample(testset, N=100)
    for epoch in range(1, epochs+1):
        # Run an epoch.
        model.train()
        train_loss = 0.0
        train_accy = 0.0
        for batch_idx, (data, target) in enumerate(trainset):
            # Zero gradient, evaluate, backward-propogate, and step the minimization.
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accy += ((output.argmax(axis=1) == target).sum() / len(target)).item()

        # Record average train loss/accuracy for this epoch.
        train_loss /= len(trainset)
        train_accy /= len(trainset)
        losses['train'].append(train_loss)
        accuracies['train'].append(train_accy)
        
        # After each epoch run on test set and print stats.
        torch.save(model.state_dict(), 'dogs.pt')
        model.eval()
        with torch.no_grad():
            test_accy = ((model(sample_data).argmax(axis=1) == sample_target).sum() / len(sample_target)).item()
            test_loss = F.nll_loss(model(sample_data), sample_target.long())
            losses['test'].append(test_loss)
            accuracies['test'].append(test_accy)
        print(f'[{epoch:<2}/{epochs}]\t train loss: {train_loss:.4f} test loss: {test_loss:.4f} train accuracy: {100*train_accy:>4.1f}% test accuracy: {100*test_accy:>4.1f}%')
    return losses, accuracies

def setup():
    # Load/crop image data per breed
    images = load_images()

    # Build train & test datasets
    trainset, testset, breeds = build_datasets(images)
    return images, trainset, testset, breeds

def load_images():
    """
    Load images into a dictionary

    returns: dict of breed name to torch.tensor of cropped image data
    """

    images = {}
    # For each breed as listed in the annotation directory
    for annotation in tqdm(os.listdir(ANNOTATION_DIR)):
        if annotation == '.DS_Store':
            continue
        # Take the name and load image files for that breed
        breed_name = annotation[annotation.find('-')+1:]
        breed_dir = '{}/{}'.format(IMG_DIR,annotation)
        fnames = os.listdir(breed_dir)

        # Torch tensor of all images for this breed.
        breed = torch.zeros(size=(len(fnames), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
        for idx, file in enumerate(fnames):
            breed[idx] = torch.tensor(crop_image('{}/{}'.format(breed_dir, file)))
        images[breed_name] = breed
    return images

def build_datasets(images, train_per_breed=80, test_per_breed=50, batch_size=64):
    """
    Turns the dictionary of images per breed into train and test datasets.
    """

    X_train = torch.zeros(size=(len(images)*train_per_breed, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
    X_test  = torch.zeros(size=(len(images)*test_per_breed, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
    y_train = torch.zeros(size=(len(images)*train_per_breed, 1), dtype=torch.long)
    y_test  = torch.zeros(size=(len(images)*test_per_breed, 1), dtype=torch.long)

    breeds = []
    for breed_idx, kv in tqdm(enumerate(images.items())):
        breed, image_arr = kv

        idx_lo_train = breed_idx*train_per_breed
        idx_lo_test  = breed_idx*test_per_breed
        indices = np.random.randint(len(image_arr),size=(train_per_breed+test_per_breed))

        X_train[idx_lo_train : idx_lo_train+train_per_breed, :,:,:] = image_arr[indices[:train_per_breed]]
        y_train[idx_lo_train : idx_lo_train + train_per_breed, 0] = breed_idx

        X_test[idx_lo_test : idx_lo_test+test_per_breed, :,:,:] = image_arr[indices[train_per_breed:]]
        y_test[idx_lo_test : idx_lo_test + test_per_breed, 0] = breed_idx

        # Record index to breed name
        breeds.append(breed)

    # Normalize data
    X_train = (X_train - 127.5) / 255
    X_test  = (X_test - 127.5) / 255

    # Reshape into pytorch image: (Channels, Height, Width)
    X_train = X_train.reshape(X_train.shape[0], IMG_CHANNEL, IMG_HEIGHT, IMG_WIDTH)
    X_test  = X_test.reshape(X_test.shape[0], IMG_CHANNEL, IMG_HEIGHT, IMG_WIDTH)

    trainset = DataLoader(TensorDataset(X_train, y_train.reshape(-1)), batch_size=batch_size, shuffle=True)
    testset  = DataLoader(TensorDataset(X_test, y_test.reshape(-1)), shuffle=True)
    return trainset, testset, breeds

def preprocess_image(path):
    """
    Read image from outside dataset.
    """
    img = read_image(path)
    img = torch.tensor(cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_AREA), dtype=torch.float32)
    img = (img - 127.5) / 255.0
    return img.reshape(IMG_CHANNEL, IMG_WIDTH, IMG_HEIGHT)

def read_image(src):
    """
    Read image data into np.array using CV2

    src: path to dog image
    returns: np.array of image data
    """

    img = cv2.imread(src)
    if img is None:
        raise FileNotFoundError
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def crop_image(path):
    """
    Load and crop image from data directory

    path: path to dog image
    returns: np.array of cropped image
    """

    # Read the image data.
    image = read_image(path)

    # Load the annotation and find the bound box in the XML.
    tree = ET.parse(os.path.join(ANNOTATION_DIR, '/'.join(path.split('/')[-2:])[:-4]))
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = root.findall('object')
    bndbox = objects[0].find('bndbox')

    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)

    # 4 : margin
    xmin = max(0, xmin - 4)
    xmax = min(width, xmax + 4)
    ymin = max(0, ymin - 4)
    ymax = min(height, ymax + 4)

    # Available width.
    w = np.min((xmax - xmin, ymax - ymin))
    w = min(w, width, height)

    if w > xmax - xmin:
        xmin = min(max(0, xmin - int((w - (xmax - xmin))/2)), width - w)
        xmax = xmin + w
    if w > ymax - ymin:
        ymin = min(max(0, ymin - int((w - (ymax - ymin))/2)), height - w)
        ymax = ymin + w

    # (height, width, channels)
    img_cropped = image[ymin:ymin+w, xmin:xmin+w, :]

    # Resize image using appropriate interp method if upsize/downsize.
    interpolation = cv2.INTER_AREA if xmax - xmin > IMG_WIDTH else cv2.INTER_CUBIC
    return cv2.resize(img_cropped, (IMG_WIDTH, IMG_HEIGHT), interpolation=interpolation)
