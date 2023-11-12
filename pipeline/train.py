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

def cross_validate(nb_cross, X_train_tot, y_train_tot, X_validation, y_validation):

    """Met en oeuvre une cross_validation et enregistre et retourne une liste de modèles selectionnés selon find_model"""

    models = []
    models_path ="/kaggle/working/models_cross/"
    for i in range(nb_cross):
        print('start nb_cross : ' + str(i) + '\n\n')
        n = len(X_train_tot)
        
        X_test = X_train_tot[i*(n//nb_cross): (i+1)*(n//nb_cross)]
        y_test = X_train_tot[i*(n//nb_cross): (i+1)*(n//nb_cross)]
        
        X_train = np.concatenate((X_train_tot[:i*(n//nb_cross)], X_train_tot[(i+1)*(n//nb_cross):]))
        y_train = np.concatenate((y_train_tot[:i*(n//nb_cross)], y_train_tot[(i+1)*(n//nb_cross):]))
        
        X_train_aug, y_train_aug = data_augmentation(X_train, y_train)
        
        y_test = np.reshape(y_test, (-1, 1))
        y_train_aug = np.reshape(y_test, (-1, 1))
        
        model = find_model(X_train_aug, y_train_aug, X_test, y_test, models_path + str(400+i))
        
        if model != None :
            models.append(model)
            model.save(models_path + str(400+ i) + '.h5')
            
    return models

def find_model(X_train, y_train, X_test, y_test, path, N = 3, epochs = 100, acc_max = 0.83):

    """Selectionne un modèle d'accuracy supérieur à acc_max sur son test set. Se donne trois tentatives.
    Pour chacune des trois tentatives, on choisit sur 100 epoch la meilleure version."""

    acc_max = 0

    for i in range(N):
        print('starting try :' + str(i))
        model = model_cnn()
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"], sample_weight_mode="temporal")
        cb_early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.02) 

        for j in range(epochs):
            model.fit(X_train, y_train, epochs=1, validation_data = (X_test, y_test) , callbacks=[cb_early_stop], batch_size=32)
            test_loss, test_acc = model.evaluate(X_test, y_test)
            if test_acc > acc_max:
                model.save(path +  '.h5')
                acc_max = test_acc

    if acc_max > 0.83 :
        return tf.keras.models.load_model(path +  '.h5')

    return None

def train(model, X_train, y_train, X_test, y_test, epochs, epoch):
    """train un modele et enregistre le meilleur."""
    
    min_test_acc = 2

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"], sample_weight_mode="temporal")
    cb_early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.02) 

    for j in range(epochs):
        model.fit(X_train, y_train, epochs=1, validation_data = (X_test, y_test) , callbacks=[cb_early_stop], batch_size=32)
        test_loss, test_acc = model.evaluate(X_test, y_test)

        if test_acc < min_test_acc:
            min_test_acc = test_acc
            model.save(path +  '.h5')
    
    return tf.keras.models.load_model(path +  '.h5')




    def total_acc(models, X_val, y_val):

        """Evalue l'accuracy moyenne des modeles sélectionnés sur le validation set"""
        final_acc = 0
        predict = []
        for model in models:
            predict.append(model.predict(X_val))
        predict = np.array(predict)
        predict = np.mean(predict, axis=0)
        predict = np.argmax(predict, axis = 1)
        
        final_acc = np.sum(predict==y_val)/len(y_val)
        return final_acc