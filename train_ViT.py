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

def evaluate(model, dataloader):
    """ Evaluates the given model on the given dataset. Prints and saves the results. """
    model.eval()
    loss_fn = torch.nn.functional.cross_entropy
    
    epoch_loss = 0
    epoch_num_correct = 0
    num_samples = 0
    
    for i, batch in enumerate(dataloader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        loss = loss_fn(preds, labels)
        epoch_loss += loss.detach().cpu().numpy() * len(images)
        epoch_num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
        num_samples += len(images)
        epoch_loss /= num_samples
        
    epoch_acc = epoch_num_correct / num_samples
    
    with open(results_path, "a") as file:
        file.write(f"Evaluation | val_loss_epoch: {epoch_loss}, val_acc: {epoch_acc} \n\n\n")
    print(f"Evaluation | val_loss_epoch: {epoch_loss}, val_acc: {epoch_acc} \n")

def train():
    global threshold
    
    loss_fn = torch.nn.functional.cross_entropy

    for iteration in tqdm(range(20)):
        # Define the model
        model = TransformerFinetune(num_classes, True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = betas)

        # Increase the threshold
        if threshold < 0.9:
            threshold += 0.03

        # Write some data
        t = datetime.datetime.now()
        with open(results_path, "a") as file:
            file.write(f"ITERATION: {iteration}\nThreshold: {threshold}\n\n")
            file.write(f"Time: {t.hour}h{t.minute}min{t.second}s\n")
            file.write(f"Pseudo-labelled: {len(dataModule.pseudo_labelled_dataset)}\n")
            file.write(f"Unlabelled: {len(dataModule.unlabelled_dataset)}\n")
        print(f"ITERATION: {iteration}")
        print(f"Threshold: {threshold}")
        print(f"Time: {t.hour}h{t.minute}min{t.second}s\n")
        print(f"Pseudo-labelled: {len(dataModule.pseudo_labelled_dataset)}")
        print(f"Unlabelled: {len(dataModule.unlabelled_dataset)}")

        model.train()
        # For each of the 5 folds:
        for i in range(5):
            model.train()

            # Get the folds
            train_loader = dataModule.train_dataloader(i % 5)
            val_loader = val_loader_list[i % 5]

            # For n_epochs // 5, train the model on the folds
            for _ in range(n_epochs // 5):

                epoch_loss = 0
                epoch_num_correct = 0
                num_samples = 0

                for j, batch in enumerate(train_loader):
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    preds = model(images)

                    # Turn the (hard) labels into smooth labels
                    epsilon = 0.05
                    labels_one_hot = torch.zeros(len(labels), num_classes).to(device)
                    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                    smooth_labels = (1 - epsilon) * labels_one_hot + epsilon / num_classes

                    # Compute the loss, backpropagate it
                    loss = loss_fn(preds, smooth_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.detach().cpu().numpy() * len(images)
                    epoch_num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                    num_samples += len(images)
                    
                epoch_loss /= num_samples
                epoch_acc = epoch_num_correct / num_samples

                # Write some data
                with open(results_path, "a") as file:
                    file.write(f"Iteration: {iteration} Epoch: {i}, train_loss_epoch: {epoch_loss}, train_acc: {epoch_acc}\n")
                print(f"Iteration: {iteration} Epoch: {i}, train_loss_epoch: {epoch_loss}, train_acc: {epoch_acc}")

            # Evaluate the model on the validation set
            evaluate(model, val_loader)

        # Find the images to be pseudo-labelled
        unlabelled_loader = dataModule.unlabelled_dataloader()
        indexes_unlabelled_to_pseudo = torch.zeros(0).to(device)
        labels_unlabelled_to_pseudo = torch.zeros(0).to(device)
        model.eval()

        for i, batch in enumerate(tqdm(unlabelled_loader)):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            soft = torch.nn.Softmax(dim = 1)(preds)

            # Select the images for which the confidence of the model is > threshold.
            unlabelled_to_pseudo = torch.where(soft > threshold)
            indexes_unlabelled_to_pseudo = torch.concat((indexes_unlabelled_to_pseudo, batch_size * i + unlabelled_to_pseudo[0]))
            labels_unlabelled_to_pseudo = torch.concat((labels_unlabelled_to_pseudo, unlabelled_to_pseudo[1]))

        dataModule.update_pseudo_labelled(indexes_unlabelled_to_pseudo, labels_unlabelled_to_pseudo)

        # Save the datasets
        with open(data_path, "wb") as file:
            pickle.dump((dataModule.dataset_list, dataModule.unlabelled_dataset, dataModule.pseudo_labelled_dataset),
                        file)

        # Save the model
        torch.save(model.state_dict(), model_path)

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
