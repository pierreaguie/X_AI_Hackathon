import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image


import numpy as np
import os
import pandas as pd

#seed everything
torch.manual_seed(0)

transform = transforms.Compose([transforms.RandomRotation(180),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])

def load_data(dataset_path, transform):
    os.chdir(dataset_path)

    images = []
    labels = []
 
    #parcours des fichiers pour load les images.
    for i in os.listdir():
        os.chdir(i)
        for filename in os.listdir():
            image = Image.open(dataset_path + '/' + i + '/' +filename)
            img = transform(image)
            images.append(img)
            if i =="plume":
                labels.append(1)
            else:
                labels.append(0)
        os.chdir("..")

    #normalizing images
    images = torch.concatenate(images)
    images = images/images.max()
    images = torch.chunk(images, images.shape[0])

    # Split the dataset into training and testing sets
    data = [(images[i], labels[i])for i in range(len(labels))]
    training_set, test_set = torch.utils.data.random_split(data, [0.7,0.3])

    training_set = DataLoader(training_set, shuffle = True, batch_size= 16)
    test_set = DataLoader(test_set, shuffle = True, batch_size = 16)

    return training_set, test_set



class DataModule:
    """Classe pour utiliser les méthodes de chargement de Pytorch, seulement génère des éléements de shape [3,64,64]
    et non [1,64,64]."""
    def __init__(
        self,
        dataset_path,
        transform,
        batch_size,
        num_workers,
    ):
        print(dataset_path)

        # Create the training dataset
        self.dataset = ImageFolder(dataset_path, transform = transform)
        
        # Split the dataset in 5
        self.dataset_list = torch.utils.data.random_split(self.dataset, [0.7, 0.3], generator = seed)

        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def val_dataloader_list(self):
        """ Returns the list containing a training DataLoader and a test Dataloader """
        val_loader_list = []
        val_loader_list.append(DataLoader(
                            self.dataset_list[0],
                            batch_size = self.batch_size,
                            shuffle = False,
                            num_workers = self.num_workers, generator = seed
                            ))
        val_loader_list.append(DataLoader(
                            self.dataset_list[1],
                            batch_size = self.batch_size,
                            shuffle = False,
                            num_workers = self.num_workers, generator = seed
                            ))
        return val_loader_list