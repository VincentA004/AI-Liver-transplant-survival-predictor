import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split

def sgd(x_train, x_val, y_train, y_val):

    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter = 100000, shuffle= True))
    clf.fit(x_train, y_train)
    predict_list = clf.predict(x_val)
    acc = metrics.accuracy_score(y_val, predict_list)
    auc = metrics.roc_auc_score(y_val, predict_list)
    cm = metrics.confusion_matrix(y_val, predict_list)

    print(cm)

    print(f'{kern} accuracy : {acc}')
    print(f'{kern} auc : {auc}')

    return acc, auc


ld = pd.read_hdf('LIVER_DATA_reduced.hdf5', key = 'df')


"""print(ld['fail'].value_counts())
print(ld.loc[ld['fail'] == 1]) """

train = ld.iloc[:75000]
val = ld.iloc[75000:]

pos_train = train.loc[train['fail'] == 1]
neg_train = train.loc[train['fail'] == 0]
neg_train_balanced = neg_train.sample(n = len(pos_train.index))

balanced_train = pd.concat([pos_train, neg_train_balanced])

y_train = balanced_train['fail'].to_numpy()
x_train = balanced_train.drop('fail', axis = 1)

y_val = val['fail'].to_numpy()
x_val = val.drop('fail', axis = 1)


"""x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.167, random_state=1) # 0.25 x 0.8 = 0.2

test_data = pd.DataFrame(x_test)['fail'] = pd.DataFrame(y_test)
test_data.to_hdf('test_data.hdf5', key = 'df')"""

kernel = ['linear']

hyperparam = {}

for kern in kernel:
    hyperparam[kern] = {}
    acc, auc = sgd(x_train, x_val, y_train, y_val)
    hyperparam[kern]['Accuracy :'] = acc
    hyperparam[kern]['AUC :'] = auc

print(hyperparam)
#pd.DataFrame(hyperparam).to_hdf('results_svm.hdf5', key = 'df')


