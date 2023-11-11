import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder, DatasetFolder

# Seed everything.
torch.manual_seed(0)

class DataModule:
    def __init__(
        self,
        train_dataset_path,
        unlabelled_dataset_path,
        train_transform,
        unlabelled_transform,
        batch_size,
        num_workers,
    ):
        print(train_dataset_path, unlabelled_dataset_path)

        # Create the training dataset
        self.train_dataset = ImageFolder(train_dataset_path, transform = train_transform)
        
        # Split the dataset in 5
        self.dataset_list = torch.utils.data.random_split(self.train_dataset, [0.2] * 5)

        # Create the unlabelled dataset
        self.unlabelled_dataset = ImageFolder(unlabelled_dataset_path, transform = unlabelled_transform)

        # Create the (empty) pseudo-labelled dataset
        self.pseudo_labelled_dataset = ImageFolder(train_dataset_path, transform = train_transform)
        self.pseudo_labelled_dataset.samples = []

        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def val_dataloader_list(self):
        """ Returns a list of 5 DataLoaders for validation """
        val_loader_list = [DataLoader(
                            x,
                            batch_size = self.batch_size,
                            shuffle = False,
                            num_workers = self.num_workers,
                            ) for x in self.dataset_list]
        return val_loader_list

    def train_dataloader(self, i):
        """
        i: indicates which train set has to be used (i is 0, 1, 2, 3 or 4)

        Returns a DataLoader containing 80% of the training set, and the pseudo-labelled images
        """
        self.concat = torch.utils.data.ConcatDataset([self.dataset_list[j] for j in range(5) if j != i]
                                                     + [self.pseudo_labelled_dataset])

        train_loader = DataLoader(
                        self.concat,
                        batch_size = self.batch_size,
                        shuffle = True,
                        num_workers = self.num_workers
                        )
        
        return train_loader
    
    def unlabelled_dataloader(self):
        """ Returns the DataLoader with all unlabelled images """
        return DataLoader(self.unlabelled_dataset,
                            batch_size = self.batch_size,
                            shuffle = False,
                            num_workers = self.num_workers)
    
    def update_pseudo_labelled(self, indexes_unlabelled_to_pseudo, labels_unlabelled_to_pseudo):
        """
        indexes_unlabelled_to_pseudo: tensor containing the indexes in self.unlabelled_dataset of images to pseudo-label
        labels_unlabelled_to_pseudo: tensor containing the corresponding labels

        Updates self.pseudo_labelled_dataset and self.unlabelled_dataset
        """
        labels_unlabelled_to_pseudo = torch.flip(labels_unlabelled_to_pseudo, (0,))
        indexes_unlabelled_to_pseudo = torch.flip(indexes_unlabelled_to_pseudo, (0,))

        self.pseudo_labelled_dataset.samples += [(self.unlabelled_dataset.samples[int(indexes_unlabelled_to_pseudo[i])][0],
                                                int(labels_unlabelled_to_pseudo[i]))
                                                for i in range(len(labels_unlabelled_to_pseudo))]

        for index in indexes_unlabelled_to_pseudo:
            self.unlabelled_dataset.samples.pop(int(index))



