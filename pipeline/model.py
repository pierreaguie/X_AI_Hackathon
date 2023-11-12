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


def model_cnn():
    return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_size[0], img_size[1], 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2() ,kernel_initializer=keras.initializers.he_normal(), bias_initializer=keras.initializers.Zeros()),
            tf.keras.layers.Dense(num_classes, activation="softmax", kernel_initializer=keras.initializers.he_normal())
        ])