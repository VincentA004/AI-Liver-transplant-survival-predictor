import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from keras import layers, models
from sklearn.feature_selection import SelectKBest, f_regression
import gc
from keras import backend as K 

min_max_scaler = MinMaxScaler()

ld = pd.read_hdf('LIVER_DATA_allwaitlist.hdf5', key = 'df')

ld = ld.sample(frac = 1)
fail = ld['fail'].to_numpy()

cols = ld.columns.drop('fail')


kb = SelectKBest(k = 8)

ld = pd.DataFrame(kb.fit_transform(ld.drop('fail', axis = 1).to_numpy(), fail))

feature_names = kb.get_feature_names_out(cols)
print(feature_names)
ld.columns = feature_names

temp = ld.columns

ld = pd.DataFrame(min_max_scaler.fit_transform(ld[temp]))
ld.columns = temp

print(temp)

ld['fail'] = fail
train = ld.iloc[:round(len(ld)*.75)]
temp = ld.iloc[round(len(ld)*.75):]
val = temp.iloc[:round(len(temp)*.60)]
test = temp.iloc[round(len(temp)*.60):]

print(train['fail'].value_counts())
print(val['fail'].value_counts())

pos_train = train.loc[train['fail'] == 1]
neg_train = train.loc[train['fail'] == 0]
neg_train_balanced = neg_train.sample(n = len(pos_train.index))

balanced_train = pd.concat([pos_train, neg_train_balanced])

y_train = balanced_train['fail'].to_numpy()
x_train = balanced_train.drop('fail', axis = 1).to_numpy()

y_val = val['fail'].to_numpy()
x_val = val.drop('fail', axis = 1).to_numpy()

y_test = test['fail'].to_numpy()
x_test = test.drop('fail', axis = 1).to_numpy()

np.savez_compressed('feature_imp_test', data = x_test, labels = y_test, feature_labels = temp)

print(x_train.shape)
print(balanced_train['fail'].value_counts())
print(val['fail'].value_counts())
print(test['fail'].value_counts())


def define_model():

    model = models.Sequential()

    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    # compile model
    model.compile(optimizer= 'adam', loss="binary_crossentropy", metrics= tf.keras.metrics.AUC(name="auc") )
    return model



cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='feature_importance.h5', 
                                                monitor = 'val_auc', 
                                                mode = 'max',
                                                save_best_only=True,
                                                verbose=1)

model = define_model()

history = model.fit(
        x_train,y_train,
        epochs = 15,
        validation_data = (x_val, y_val),
        callbacks=[cp_callback])

score = model.evaluate(x_test,y_test)

print(f'AUC: {score[1]}')





