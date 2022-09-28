import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

features = pd.read_csv('All_features.csv')

feature_list = features.values.tolist()

ld = pd.read_hdf('LIVER_DATA_v2.hdf5', key = 'df')

ld_reduced = pd.DataFrame()

for feature in feature_list:
    ld_reduced[feature[0]] = ld[feature[0]]

cols = ld_reduced.columns[ld_reduced.isnull().mean()>0.75]
ld_reduced.drop(cols, axis=1)

cols = ld_reduced.columns


imp = IterativeImputer(max_iter = 20)
ld_reduced = pd.DataFrame(imp.fit_transform(ld_reduced))
ld_reduced.columns = cols

print(ld_reduced)

ld_reduced.to_hdf('LIVER_DATA_reduced.hdf5', key = 'df')

#ld = pd.read_hdf('LIVER_DATA_reduced.hdf5', key = 'df')

ld_reduced.to_csv('dataset.csv')

#for col in ld.select_dtypes(include = 'number').columns:


