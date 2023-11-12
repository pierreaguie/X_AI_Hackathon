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

from data import *
from model import *
from train import *


dataset_path = "/kaggle/input/qb-3-axel/data"
metadata_path = "/kaggle/input/qb-3-axel/data/metadata.csv"
model_path = 'model'
num_fold = 6

def retrain_cross_validate():
    """Met en place une sélection de modèle en utilisant la cross validation"""

    images, labels = load_data(dataset_path, metadata_path)
    X_train_global, X_validation, y_train_global, y_validation,  y_validation_sc = validation_split(images, labels)
    models = cross_validate(num_fold, X_train_global, y_train_global, X_validation, y_validation)
    return models

def retrain():

    """" Train un modèle sur un dataset et garde la meilleure version sur les différentes epochs"""

    images, labels = load_data(dataset_path, metadata_path)
    X_train, X_test, y_train, y_test,  y_test_sc = validation_split(images, labels)
    model = model_cnn()
    return train(model, X_train, y_train, path, epoch)