from sklearn import metrics
import pandas as pd
import numpy as np

ld = pd.read_hdf('LIVER_DATA_reduced.hdf5', key = 'df')
ld_y = pd.read_hdf('LIVER_DATA_v3.hdf5', key = 'df')


print(ld_y['fail'].value_counts())
