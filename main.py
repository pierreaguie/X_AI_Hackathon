from models import *
from data import *
from train_ViT import *
from parameters import *

if __name__ == "__main__":

    if act == "train":
        
        dataModule = DataModule(train_path,
                        unlabelled_path,
                        transform_augmentation,
                        transform,
                        batch_size,
                        num_workers)

        val_loader_list = dataModule.val_dataloader_list()