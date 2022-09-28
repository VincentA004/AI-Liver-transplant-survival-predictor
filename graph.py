import pandas as pd

data=pd.read_hdf('feature_auc_curve.hdf5', key = 'df')

print(data)
