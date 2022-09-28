import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.feature_selection import SelectKBest, f_regression
import gc
from keras import backend as K 
from sklearn import metrics

#x_test = np.load('feature_imp_test.npz')['data']
#y_test = np.load('feature_imp_test.npz')['labels']

def softmax(data):
    f = []
    for x in data:
        if x == 1:
            f.append([0,1])
        elif x == 0:
            f.append([1,0])
    fail = np.array(f)
    return fail

min_max_scaler = MinMaxScaler()

ld = pd.read_hdf('LIVER_DATA_reduced.hdf5', key = 'df')

ld = ld.sample(frac = 1)
fail = ld['fail'].to_numpy()


cols = ld.columns.drop('fail')


kb = SelectKBest(k = 50)

ld = pd.DataFrame(kb.fit_transform(ld.drop('fail', axis = 1).to_numpy(), fail))

feature_names = kb.get_feature_names_out(cols)
print(feature_names)
ld.columns = feature_names

temp = ld.columns

ld = pd.DataFrame(min_max_scaler.fit_transform(ld[temp]))
ld.columns = temp

print(temp)

ld['fail'] = fail

y_test = softmax(ld['fail'])

x_test = ld.drop('fail', axis = 1).to_numpy()

def define_model():

    model = models.Sequential()
    model.add(layers.Dense(4096, activation='relu', input_shape = (None, 50)))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    # compile model
    model.compile(optimizer= 'adam', loss="categorical_crossentropy", metrics= tf.keras.metrics.AUC(name="auc") )
    return model

model = define_model()

model.load_weights('feature_importance.h5')

score = model.evaluate(x_test, y_test)