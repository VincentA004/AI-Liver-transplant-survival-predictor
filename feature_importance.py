import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from keras import layers, models
from sklearn.feature_selection import SelectKBest, f_regression
import gc
from keras import backend as K 

AUC = .9071


def define_model():

    model = models.Sequential()

    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile model
    model.compile(optimizer= 'adam', loss="binary_crossentropy", metrics= tf.keras.metrics.AUC(name="auc") )
    return model

model = define_model()
model.load_weights()

score = model.evaluate(x_test,y_test)

print(f'AUC: {score[1]}')
