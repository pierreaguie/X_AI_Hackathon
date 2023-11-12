import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image

from tqdm import tqdm
import math

import numpy as np
import os
import pandas as pd


def eval(model, device, dataloader):
    """fonction qui évalue la CrossEntropyLoss et l'accuracy du modele sur le dataloader."""
    model.to(device)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    epoch_loss = 0
    epoch_num_correct = 0
    num_samples = 0

    for i, batch in enumerate(dataloader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        loss = loss_fn(preds, labels)

        #on somme les loss et on compte les bon résultats
        epoch_loss += loss.detach().cpu().numpy() * len(images)
        epoch_num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
        num_samples += len(images)

    #on moyenne
    epoch_loss /= num_samples
    epoch_acc = epoch_num_correct / num_samples
    return epoch_loss, epoch_acc


def train_plot_all(model, device, dataloader, test_set, epoch, rate = 1e-3, path = path):
    """Fonction train qui retourne la liste des loss et accuracy (training et test) au cours des epochs.
    On utilise SGD comme optimizer.
    Enregistre le modèle à l'époque où il a la meilleure loss"""
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    acc = 0
    min_test_loss = 10
    
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
        
        test_loss, test_acc = eval(model, device, test_set)
        train_loss, train_acc = eval(model, device, training_set)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if t%20 == 0:
            if test_loss < min_test_loss:
                acc = test_acc
                min_test_loss = test_loss
                torch.save(model.state_dict(),path + '.pth')
            print(f'test loss {test_loss}, train_loss {train_loss}')
        
    
    
    return train_losses, test_losses, train_accuracies, test_accuracies, min_test_loss, acc

def train(model, device, dataloader, epoch, rate = 1e-3):
    """Ici on train avec Adam.
    On ne fait qu'entrainer sur le dataloader et on affiche l'accuracy sur le test set de temps en temps."""
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    for t in tqdm(range(epoch)):
        for i, (input_data, target) in enumerate(dataloader):
            input_data, target = input_data.to(device), target.to(device)
            y_pred = model(input_data)
            loss = loss_fn(y_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(loss.detach().item())
        l,a = eval(model, device, test_set)
        #print(loss.detach().item())
        if t%50:
            print('acc', a)
