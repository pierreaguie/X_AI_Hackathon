import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import utils
import keras
import imageio
from sklearn.utils import shuffle

def add_image(images, img, labels, label):
    """ajoute img à la liste images et le label corespondants à labels"""

    images.append(img)
    if label=="yes":
        labels.append(1)
    else:
        labels.append(0)
    return 


def load_data(dataset_path, metadata_path):
    """retourne les listes d'images et de labels"""

    img_size = (64, 64)
    num_classes = 2

    # Load the dataset of images and their corresponding labels
    images_or = []
    labels_or = []

    metadata = pd.read_csv("/kaggle/input/qb-3-axel/data/metadata.csv")

    for i in range(len(metadata)):
        lign = metadata.iloc[i]
        filename = lign["path"] + '.tif'
        
        img = load_img(os.path.join(dataset_path, filename), target_size=img_size, color_mode="grayscale")
        img = img_to_array(img)
        add_image(images_or, img, labels_or, lign["plume"])
    return images_or, labels_or

def add_image_bis(images, img, labels, label):
    """ajoute une image à une liste et le label correspondant"""
    images.append(img)
    if label[0]==1:
        labels.append(1)
    else:
        labels.append(0)
    return 


def data_augmentation(X_train , y_train ):
    """Effectue la data augmentation. Effectue 6 opérations différentes"""

    X_train_aug = []
    y_train_aug = []
    for i in range(len(X_train)):
        img = X_train[i]
        add_image_bis(X_train_aug, img, y_train_aug, y_train[i])
        
        img = np.copy(np.fliplr(img))
        add_image_bis(X_train_aug, img, y_train_aug, y_train[i])

        img = np.copy(np.flipud(img))
        add_image_bis(X_train_aug, img, y_train_aug, y_train[i])

        img = np.copy(np.rot90(img))
        add_image_bis(X_train_aug, img, y_train_aug, y_train[i])

        img = np.copy(np.rot90(np.rot90(img)))
        add_image_bis(X_train_aug, img, y_train_aug, y_train[i])

        img = np.copy(np.rot90(np.rot90(np.rot90(img))))
        add_image_bis(X_train_aug, img, y_train_aug, y_train[i])

        img = np.copy(np.transpose(img, (1, 0, 2)))
        add_image_bis(X_train_aug, img, y_train_aug, y_train[i])
        
    return X_train_aug, y_train_aug

def validation_split(images, labels):
    X_train_global, X_validation, y_train_global, y_validation = train_test_split(images_or, labels_or, test_size=0.2, random_state = 1)
    y_train_global = np.reshape(y_train_global, (-1,1))
    y_validation = np.reshape(y_validation, (-1,1))
    y_validation_sc = y_validation[:,0]

    return X_train_global, X_validation, y_train_global, y_validation,  y_validation_sc