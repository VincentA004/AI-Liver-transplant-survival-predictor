import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from keras import layers, models
from sklearn.feature_selection import SelectKBest, f_regression
import gc
from keras import backend as K 
from sklearn import metrics
import shap
import matplotlib.pyplot as plt


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

ld = pd.read_hdf('LIVER_DATA_reduced_updated_2.hdf5', key = 'df')

ld = ld.sample(frac = 1)
fail = ld['fail'].to_numpy()


cols = ld.columns.drop('fail')


kb = SelectKBest(k = 2)

ld = pd.DataFrame(kb.fit_transform(ld.drop('fail', axis = 1).to_numpy(), fail))

feature_names = kb.get_feature_names_out(cols)
print(feature_names)
ld.columns = feature_names

temp = ld.columns

ld = pd.DataFrame(min_max_scaler.fit_transform(ld[temp]))
ld.columns = temp

f_n = temp.to_list()

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

neg_train = neg_train.drop(neg_train_balanced.index)

balanced_train = pd.concat([pos_train, neg_train_balanced])
test = test.append(neg_train)


y_train = softmax(balanced_train['fail'])
print(y_train)
x_train = balanced_train.drop('fail', axis = 1).to_numpy()

y_val = softmax(val['fail'])
x_val = val.drop('fail', axis = 1).to_numpy()

y_test = softmax(test['fail'])
x_test = test.drop('fail', axis = 1).to_numpy()

print(x_train.shape)
print(x_test.shape)

np.savez_compressed('feature_imp_test', data = x_test, labels = y_test, feature_labels = temp)
np.savez_compressed('feature_imp_train', data = x_train, labels = y_train, feature_labels = temp)

print(x_train.shape)
print(balanced_train['fail'].value_counts())
print(val['fail'].value_counts())
print(test['fail'].value_counts())


def define_model():

    model = models.Sequential()
    model.add(layers.Dense(4096, activation='relu', input_shape = (None, 2)))
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


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='feature_importance.h5', 
                                                monitor = 'val_auc', 
                                                mode = 'max',
                                                save_best_only=True,
                                                verbose=1)

model = define_model()

history = model.fit(
        x_train,y_train,
        epochs = 25,
        validation_data = (x_val, y_val),
        callbacks=[cp_callback])

"""plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])"""

plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])

plt.show()

model2 = define_model()
model2.load_weights('feature_importance.h5')
score = model2.evaluate(x_test, y_test)
print(score[1])

print(feature_names)

"""explainer = shap.KernelExplainer(model2, x_train[:100])
shap_values = explainer.shap_values(x_train[:100])
print(shap_values)
shap.summary_plot(shap_values[1], feature_names = f_n)"""









