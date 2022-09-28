from inspect import CO_NEWLOCALS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

def svm(x_train, x_val, y_train, y_val, num):

    clf = make_pipeline(StandardScaler(), SVC(kernel = 'poly', gamma='auto'))
    clf.fit(x_train, y_train)
    predict_list = clf.predict(x_val)
    acc = metrics.accuracy_score(y_val, predict_list)
    auc = metrics.roc_auc_score(y_val, predict_list)

    print(f'{num} feature accuracy : {acc}')
    print(f'{num} feature auc : {auc}')

    return acc, auc


feature_param = {}

for num in range(198, 49, -2):
    ld = pd.read_hdf('LIVER_DATA_reduced.hdf5', key = 'df')

    cols = ld.columns.drop('fail')

    kb = SelectKBest(k = num)
    fail = ld['fail'].to_numpy()
    ld = pd.DataFrame(kb.fit_transform(ld.drop('fail', axis = 1).to_numpy(), fail))

    ld.columns = kb.get_feature_names_out(cols)
    ld['fail'] = fail
    train = ld.iloc[:75000]
    val = ld.iloc[75000:]

    pos_train = ld.loc[ld['fail'] == 1]
    neg_train = ld.loc[ld['fail'] == 0]
    neg_train_balanced = neg_train.sample(n = len(pos_train.index))

    balanced_train = pd.concat([pos_train, neg_train_balanced])

    y_train = balanced_train['fail'].to_numpy()
    x_train = balanced_train.drop('fail', axis = 1).to_numpy()

    y_val = val['fail'].to_numpy()
    x_val = val.drop('fail', axis = 1).to_numpy()

    feature_param[f'{num} features'] = {}

    acc, auc = svm(x_train, x_val, y_train, y_val, num)
    feature_param[f'{num} features']['Accuracy :'] = acc
    feature_param[f'{num} features']['AUC :'] = auc

print(feature_param)
pd.DataFrame(feature_param).to_hdf('feature_param_svm.hdf5', key = 'df')


