import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder, DatasetFolder

from tqdm import tqdm

import numpy as np

import datetime

import pickle

from data import *
from models import *
from paths import *

# Seed everything.
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, device, dataloader, epoch, rate = 1e-4):

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=rate)

    loss_fn = nn.CrossEntropyLoss()
    train_losses = []

    for t in tqdm(range(epoch)):
        for i, (input_data, target) in enumerate(dataloader):

            input_data, target = input_data.to(device), target.to(device)

            #on calcule la prediction 
            y_pred = model(input_data)
            loss = loss_fn(y_pred, target)
            optimizer.zero_grad()
            #back propagation
            loss.backward()
            optimizer.step()

        train_losses.append(loss.detach().item())

    return train_losses


# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

# Transform with data augmentation
transform_augmentation = transforms.Compose([
    transforms.autoaugment.AutoAugment(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

# Number of classes
num_classes = 48

# Parameters
lr = 1e-3
betas = [0.9, 0.999]
batch_size = 128
threshold = 0.7
n_epochs = 15
num_workers = 0

# Paths to files where results are stored
results_path = "results_ViT.txt"
model_path = "model_ViT.pth"
data_path = "data_ViT.pth"

# Initialize all the datasets
dataModule = DataModule(train_path,
                        unlabelled_path,
                        transform_augmentation,
                        transform,
                        batch_size,
                        num_workers)

val_loader_list = dataModule.val_dataloader_list()

# Train the model
train()
