import pandas as pd
import datetime 
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import datasets, layers, models
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

"""
ld = pd.read_stata(f'LIVER_DATA.DTA')

ld = ld[(ld.WLHR == '') & 
        (ld.WLIN == '') & 
        (ld.WLKP == '') &
        (ld.WLLU == '') &
        (ld.WLPA == '') &
        (ld.TXHRT == '') &
        (ld.TXKID == '') &  
        (ld.TXINT == '') &
        (ld.TXLNG == '') &
        (ld.TXPAN == '') &
        (ld.INIT_AGE > 18) &
        (ld.INIT_DATE > datetime.datetime(2002, 3, 1)) &
        (ld.PX_STAT != "") &
        (pd.isna(ld.LIV_DON_TY))]

ld['ABO_A'] = np.where(ld['ABO'] == 'A', 1, 0)
ld['ABO_B'] = np.where(ld['ABO'] == 'B', 1, 0)
ld['ABO_AB'] = np.where(ld['ABO'] == 'AB', 1, 0)
ld['ABO_O'] = np.where(ld['ABO'] == 'O', 1, 0)
ld['ABO_DON_A'] = np.where(ld['ABO_DON'] == 'A', 1, 0)
ld['ABO_DON_B'] = np.where(ld['ABO_DON'] == 'B', 1, 0)
ld['ABO_DON_AB'] = np.where(ld['ABO_DON'] == 'AB', 1, 0)
ld['ABO_DON_O'] = np.where(ld['ABO_DON'] == 'O', 1, 0)

ld['COD_CAD_DON_1'] = np.where(ld['COD_CAD_DON'] == 1, 1, 0)
ld['COD_CAD_DON_2'] = np.where(ld['COD_CAD_DON'] == 2, 1, 0)
ld['COD_CAD_DON_3'] = np.where(ld['COD_CAD_DON'] == 3, 1, 0)
ld['COD_CAD_DON_4'] = np.where(ld['COD_CAD_DON'] == 4, 1, 0)


ld['CORONARY1'] = np.where(ld['CORONARY_ANGIO_DON'] == "N", 0, np.where(ld['CORONARY_ANGIO_NORM_DON'] == "N", 1, 2))
ld['DEATH_CIRCUM_DON'] = np.where(ld['DEATH_CIRCUM_DON'] == 6, 1, 0)
ld['DEATH_MECH_DON'] = np.where(ld['DEATH_MECH_DON'] == 12, 1, 0)

ld['DGN_TCR_AHN'] = np.where(ld['DGN_TCR'] == (4100 | 4101 | 4102 | 4103 | 4104 | 4105 | 4106 | 4107 | 4108 | 4110 | 4217) , 1, 0)
ld['DGN_TCR_AUTOIMMUNE'] = np.where(ld['DGN_TCR'] == 4212 , 1, 0)
ld['DGN_TCR_CRYPTOGENIC'] = np.where(ld['DGN_TCR'] == (4213 | 4208) , 1, 0)
ld['DGN_TCR_ETOH'] = np.where(ld['DGN_TCR'] == 4215 , 1, 0)
ld['DGN_TCR_ETOH_HCV'] = np.where(ld['DGN_TCR'] == 4216 , 1, 0)
ld['DGN_TCR_HBV'] = np.where(ld['DGN_TCR'] == (4202 | 4592) , 1, 0)
ld['DGN_TCR_HCC'] = np.where(ld['DGN_TCR'] == (4400 | 4401 | 4402) , 1, 0)
ld['DGN_TCR_HCV'] = np.where(ld['DGN_TCR'] == (4204 | 4593) , 1, 0)
ld['DGN_TCR_NASH'] = np.where(ld['DGN_TCR'] == (4214) , 1, 0)
ld['DGN_TCR_PBC'] = np.where(ld['DGN_TCR'] == (4220) , 1, 0)
ld['DGN_TCR_PSC'] = np.where(ld['DGN_TCR'] == (4240 | 4241 | 4242 | 4245) , 1, 0)

ld['DGN2_TCR_AHN'] = np.where(ld['DGN2_TCR'] == (4100 | 4101 | 4102 | 4103 | 4104 | 4105 | 4106 | 4107 | 4108 | 4110 | 4217) , 1, 0)
ld['DGN2_TCR_AUTOIMMUNE'] = np.where(ld['DGN2_TCR'] == 4212 , 1, 0)
ld['DGN2_TCR_CRYPTOGENIC'] = np.where(ld['DGN2_TCR'] == (4213 | 4208) , 1, 0)
ld['DGN2_TCR_ETOH'] = np.where(ld['DGN2_TCR'] == 4215 , 1, 0)
ld['DGN2_TCR_ETOH_HCV'] = np.where(ld['DGN2_TCR'] == 4216 , 1, 0)
ld['DGN2_TCR_HBV'] = np.where(ld['DGN2_TCR'] == (4202 | 4592) , 1, 0)
ld['DGN2_TCR_HCC'] = np.where(ld['DGN2_TCR'] == (4400 | 4401 | 4402) , 1, 0)
ld['DGN2_TCR_HCV'] = np.where(ld['DGN2_TCR'] == (4204 | 4593) , 1, 0)
ld['DGN2_TCR_NASH'] = np.where(ld['DGN2_TCR'] == (4214) , 1, 0)
ld['DGN2_TCR_PBC'] = np.where(ld['DGN2_TCR'] == (4220) , 1, 0)
ld['DGN2_TCR_PSC'] = np.where(ld['DGN2_TCR'] == (4240 | 4241 | 4242 | 4245) , 1, 0)

ld['DIAG_AHN'] = np.where(ld['DIAG'] == (4100 | 4101 | 4102 | 4103 | 4104 | 4105 | 4106 | 4107 | 4108 | 4110 | 4217) , 1, 0)
ld['DIAG_AUTOIMMUNE'] = np.where(ld['DIAG'] == 4212 , 1, 0)
ld['DIAG_CRYPTOGENIC'] = np.where(ld['DIAG'] == (4213 | 4208) , 1, 0)
ld['DIAG_ETOH'] = np.where(ld['DIAG'] == 4215 , 1, 0)
ld['DIAG_ETOH_HCV'] = np.where(ld['DIAG'] == 4216 , 1, 0)
ld['DIAG_HBV'] = np.where(ld['DIAG'] == (4202 | 4592) , 1, 0)
ld['DIAG_HCC'] = np.where(ld['DIAG'] == (4400 | 4401 | 4402) , 1, 0)
ld['DIAG_HCV'] = np.where(ld['DIAG'] == (4204 | 4593) , 1, 0)
ld['DIAG_NASH'] = np.where(ld['DIAG'] == (4214) , 1, 0)
ld['DIAG_PBC'] = np.where(ld['DIAG'] == (4220) , 1, 0)
ld['DIAG_PSC'] = np.where(ld['DIAG'] == (4240 | 4241 | 4242 | 4245) , 1, 0)

ld['DIAB'] = np.where(ld['DIAB'] == "No", 0, 1)

values = ['CMV_IGG','CMV_IGM','CMV_STATUS','EBV_SEROSTATUS','HBV_CORE', 'HBV_SUR_ANTIGEN', 'CMV_DON', 'EBV_IGG_CAD_DON', 'EBV_IGM_CAD_DON', 'HBV_CORE_DON', 'HBV_SUR_ANTIGEN_DON', 'HCV_SEROSTATUS', 'HEP_C_ANTI_DON']

for value in values:
    ld[value] = np.where(ld[value] == 'P', 2, np.where(ld[value] == 'N', 1, 0))

for num in range(1,6):
    ld[f'ETHCAT_{num}'] = np.where(ld['ETHCAT'] == num, 1, 0)
    ld[f'ETHCAT_DON_{num}'] = np.where(ld['ETHCAT_DON'] == num, 1, 0)

ld[f'ETHCAT_OTHER'] = np.where(ld['ETHCAT'] != (1 | 2 | 3 | 4 | 5), 1, 0)
ld[f'ETHCAT_DON_OTHER'] = np.where(ld['ETHCAT_DON'] != (1 | 2 | 3 | 4 | 5), 1, 0)



ld[f'EXC_DIAG_ID_CAT1'] = np.where(ld['EXC_DIAG_ID'] == (1 | 3 | 10), 1, 0 )
ld[f'EXC_DIAG_ID_CAT2'] = np.where(ld['EXC_DIAG_ID'] == 2, 1, 0 )
ld[f'EXC_DIAG_ID_CAT3'] = np.where(ld['EXC_DIAG_ID'] == 4, 1, 0 )
ld[f'EXC_DIAG_ID_CAT4'] = np.where(ld['EXC_DIAG_ID'] == 5, 1, 0 )
ld[f'EXC_DIAG_ID_CAT5'] = np.where(ld['EXC_DIAG_ID'] == (6 | 7 | 12), 1, 0 )
ld[f'EXC_DIAG_ID_CAT6'] = np.where(ld['EXC_DIAG_ID'] == 11, 1, 0 )
ld[f'EXC_DIAG_ID_CAT7'] = np.where(ld['EXC_DIAG_ID'] == 9, 1, 0 )

ld['FUNC_STAT_TCR'] = np.where(ld['FUNC_STAT_TCR'] == (1| 2080 | 2090 | 2100), 0, np.where(ld['FUNC_STAT_TCR'] == (2 | 2040 | 2050 | 2060 | 2070), 1, 2))
ld['FUNC_STAT_TRR'] = np.where(ld['FUNC_STAT_TRR'] == (1| 2080 | 2090 | 2100), 0, np.where(ld['FUNC_STAT_TRR'] == (2 | 2040 | 2050 | 2060 | 2070), 1, 2))

ld[f'MELD_DIFF_REASON_CD_1'] = np.where(ld['MELD_DIFF_REASON_CD'] == (15 | 16), 1, 0 )
ld[f'MELD_DIFF_REASON_CD_2'] = np.where(ld['MELD_DIFF_REASON_CD'] == 8, 1, 0 )

ld['MACRO_FAT_LI_DON'] = np.where(ld['LI_BIOPSY'] == "N", 0, np.where(ld['MACRO_FAT_LI_DON'] < 30, 1, 2))

ld['CITIZENSHIP'] = np.where(ld['CITIZENSHIP'] == "U.S Citizen", 1, 0)
ld['CITIZENSHIP_DON'] = np.where(ld['CITIZENSHIP_DON'] == "U.S Citizen", 1, 0)

ld['END_STAT'] = np.where(ld['END_STAT'] == (6010 | 6011), 1, 0)
ld['INIT_STAT'] = np.where(ld['INIT_STAT'] == (6010 | 6011), 1, 0)

ld['INSULIN_DEP_DON'] = np.where(ld['INSULIN_DEP_DON'] == "No", 0, 1)

ld['LITYP'] = np.where(ld['LITYP'] == 20, 1, 0)

ld['MALIG_TYPE'] = np.where(ld['MALIG_TYPE'] == (4096 | 8192), 1, 0)

ld['PRI_PAYMENT_TRR'] = np.where(ld['PRI_PAYMENT_TRR'] == 1, 1, 0)

ld['SHARE_TY'] = np.where(ld['SHARE_TY'] == (3 | 4), 0, 1)

ld['EXC_HCC'] = np.where(ld['EXC_HCC'] == 'HCC', 1, 0)

ld['EXC_CASE'] = np.where(ld['EXC_CASE'] == 'Yes', 1, 0)

ld['GENDER'] = np.where(ld['GENDER'] == 'M', 1, 0)

ld['GENDER_DON'] = np.where(ld['GENDER_DON'] == 'M', 1, 0)

ld['EDUCATION'] = ld['EDUCATION'].replace(998,0)

ld['fail'] = np.where((ld['PX_STAT'] == 'D') & (ld['PTIME'] <= 90), 1, 0)


values = ['PORTAL_VEIN_TCR', 'PORTAL_VEIN_TRR',	'PREV_AB_SURG_TCR',	'PREV_AB_SURG_TRR',	'PREV_TX', 'MALIG_TCR', 'NON_HRT_DON', 'MALIG', 'LIFE_SUP_TRR', 'LIFE_SUP_TCR', 
          'INSULIN_DON', 'INOTROP_SUPPORT_DON', 'INIT_DIALYSIS_PRIOR_WEEK', 'HISTORY_MI_DON', 'HIST_OTH_DRUG_DON', 'HIST_INSULIN_DEP_DON', 'HIST_CIG_DON', 'HIST_CANCER_DON', 'HEPARIN_DON',
          'DIAL_TX', 'DIABETES_DON', 'DDAVP_DON', 'CLIN_INFECT_DON', 'CARDARREST_NEURO', 'BACT_PERIT_TCR', 'ANTIHYPE_DON', 'PROTEIN_URINE',	'PT_DIURETICS_DON',	'PT_OTH_DON',	'PT_STEROIDS_DON',	'PT_T3_DON',
           'PT_T4_DON',	'RECOV_OUT_US', 'TATTOOS', 'TIPSS_TCR',	'TIPSS_TRR','VASODIL_DON','VDRL_DON', 'WORK_INCOME_TCR', 'WORK_INCOME_TRR','HIST_COCAINE_DON', 'CDC_RISK_HIV_DON', 'ARGININE_DON']

for value in values:
    ld[value] = np.where(ld[value] == 'Y', 1, 0)

ld.to_hdf('LIVER_DATA_v2.hdf5', key = 'df')


ld['failure'] = np.where(ld['PX_STAT'] == 'D', 1, 0)
ld['fail'] = np.where((ld['PX_STAT'] == 'D') & (ld['PTIME'] <= 90), 1, 0)
ld['aboi'] = np.where(ld['ABO_MAT'] == "3", 1, 0)
ld['aboc'] = np.where(ld['ABO_MAT'] == "2", 1, 0)
ld['ascites'] = np.where(((ld['ASCITES_TRR_OLD'] == 'Y') | (ld['ASCITES_TX'] == 2) | (ld['ASCITES_TX'] == 3)), 1, 0)
ld['sbp_mv'] = np.where(ld['BACT_PERIT_TCR'] == "Y", 1, 0)
ld['dcd'] = np.where(ld['CONTROLLED_DON'] == "Y", 1, 0)
ld['hcv_mv'] = np.where(((ld['DIAG'] == 4204) | (ld['DIAG'] == 4206) | (ld['DIAG'] == 4216) | (ld['DIAG'] == 4593)), 1, 0)
ld['hd'] = np.where(ld['DIAL_TX'] == "Y", 1, 0)
ld['dropout'] = np.where(ld['EDUCATION'] <= 2, 1, 0)
ld['highschool'] = np.where(ld['EDUCATION'] == 3, 1, 0)
ld['technical'] = np.where(ld['EDUCATION'] == 4, 1, 0)
ld['bachelors'] = np.where(ld['EDUCATION'] == 5, 1, 0)
ld['doctor'] = np.where(ld['EDUCATION'] == 6, 1, 0)
ld['enceph'] = np.where(((ld['ENCEPH_TRR_OLD'] == 'Y') | (ld['ENCEPH_TX'] == 3)), 1, 0)
ld['aa'] = np.where(ld['ETHCAT'] == 2, 1, 0)
ld['aa_don'] = np.where(ld['ETHCAT_DON'] == 2, 1, 0)
ld['hcc'] = np.where(ld['EXC_HCC'] == 'HCC', 1, 0)
ld['fun10'] = np.where(ld['FUNC_STAT_TRR'] == 2010, 1, 0)
ld['fun20'] = np.where(ld['FUNC_STAT_TRR'] == 2020, 1, 0)
ld['fun50'] = np.where(ld['FUNC_STAT_TRR'] == 2050, 1, 0)
ld['fun60'] = np.where(ld['FUNC_STAT_TRR'] == 2060, 1, 0)
ld['fun70'] = np.where(ld['FUNC_STAT_TRR'] == 2070, 1, 0)
ld['fun80'] = np.where(ld['FUNC_STAT_TRR'] == 2080, 1, 0)
ld['fun90'] = np.where(ld['FUNC_STAT_TRR'] == 2090, 1, 0)
ld['fun100'] = np.where(ld['FUNC_STAT_TRR'] == 2100, 1, 0)
ld['hcv_don'] = np.where(ld['HEP_C_ANTI_DON'] == "P", 1, 0)
ld['height_diff'] = ld['HGT_CM_TCR'] - ld['HGT_CM_DON_CALC']
ld['lifesup'] = np.where(ld['LIFE_SUP_TRR'] == "Y", 1, 0)
ld['icu'] = np.where(ld['MED_COND_TRR'] == 1, 1, 0)
ld['hospital'] = np.where(ld['MED_COND_TRR'] == 2, 1, 0)
ld['prev_tx1'] = np.where(ld['NUM_PREV_TX'] == 1, 1, 0)
ld['prev_tx2'] = np.where(ld['NUM_PREV_TX'] == 2, 1, 0)
ld['tx3'] = np.where(ld['NUM_PREV_TX'] >= 3, 1, 0)
ld['vent'] = np.where(ld['ON_VENT_TRR'] == 1, 1, 0)
ld['porbleed_week'] = np.where(ld['PORTAL_VEIN_TCR'] == "Y", 1, 0)
ld['portal'] = np.where(ld['PORTAL_VEIN_TRR'] == "Y", 1, 0) 
ld['private'] = np.where(ld['PRI_PAYMENT_TRR'] == 1, 1, 0) 
ld['medicaid'] = np.where(ld['PRI_PAYMENT_TRR'] == 2, 1, 0)
ld['prev_surg'] = np.where(ld['PREV_AB_SURG_TRR'] == "Y", 1, 0) 
ld['region1'] = np.where(ld['REGION'] == 1, 1, 0)
ld['region2'] = np.where(ld['REGION'] == 2, 1, 0)
ld['region3'] = np.where(ld['REGION'] == 3, 1, 0)
ld['region4'] = np.where(ld['REGION'] == 4, 1, 0)
ld['region5'] = np.where(ld['REGION'] == 5, 1, 0)
ld['region6'] = np.where(ld['REGION'] == 6, 1, 0)
ld['region7'] = np.where(ld['REGION'] == 7, 1, 0)
ld['region8'] = np.where(ld['REGION'] == 8, 1, 0)
ld['region9'] = np.where(ld['REGION'] == 9, 1, 0)
ld['region10'] = np.where(ld['REGION'] == 10, 1, 0)
ld['region11'] = np.where(ld['REGION'] == 11, 1, 0)
ld['regional'] = np.where(ld['SHARE_TY'] == 4, 1, 0)
ld['national'] = np.where(ld['SHARE_TY'] == 5, 1, 0)
ld['foreign'] = np.where(ld['SHARE_TY'] == 6, 1, 0)
ld['tip'] = np.where(ld['TIPSS_TRR'] == 'Y', 1, 0 )
ld['weight_diff'] = ld['WGT_KG_TCR'] - ld['WGT_KG_DON_CALC']
ld['work'] = np.where(ld['WORK_INCOME_TRR'] == 'Y', 1, 0 )

ld[['AGE','AGE_DON','ALBUMIN_TX','BMI_TCR','COLD_ISCH','CREAT_DON','DISTANCE','FINAL_MELD_PELD_LAB_SCORE',
    'FINAL_SERUM_SODIUM', 'height_diff', 'INR_TX', 'PH_DON', 'SGOT_DON', 'TBILI_TX', 'TBILI_DON', 'WARM_ISCH_TM_DON', 'weight_diff']] = min_max_scaler.fit_transform(ld[['AGE','AGE_DON','ALBUMIN_TX','BMI_TCR','COLD_ISCH','CREAT_DON','DISTANCE','FINAL_MELD_PELD_LAB_SCORE',
    'FINAL_SERUM_SODIUM', 'height_diff', 'INR_TX', 'PH_DON', 'SGOT_DON', 'TBILI_TX', 'TBILI_DON', 'WARM_ISCH_TM_DON', 'weight_diff']])


ld.to_hdf('LIVER_DATA_v3.hdf5', key = 'df')

features = ['AGE','AGE_DON','ALBUMIN_TX','BMI_TCR','COLD_ISCH','CREAT_DON','DISTANCE','FINAL_MELD_PELD_LAB_SCORE',
    'FINAL_SERUM_SODIUM', 'height_diff', 'INR_TX', 'PH_DON', 'SGOT_DON', 'TBILI_TX', 'TBILI_DON', 'WARM_ISCH_TM_DON', 'weight_diff', 'aboi', 'aboc', 'ascites', 
    'sbp_mv', 'dcd', 'hcv_mv', 'hd', 'dropout', 'highschool', 'technical', 'bachelors', 'doctor', 'enceph', 'aa', 'aa_don', 'hcc', 'fun10', 'fun20',  'fun50', 
    'fun60', 'fun70', 'fun80', 'fun90', 'fun100','hcv_don', 'lifesup', 'icu','hospital', 'prev_tx1', 'prev_tx2', 'tx3', 'vent', 'porbleed_week', 'portal',
    'private', 'medicaid', 'prev_surg', 'region1', 'region2', 'region3', 'region4', 'region5', 'region6', 'region7', 'region8', 'region9', 'region10', 'region11', 
    'regional', 'national', 'foreign', 'tip', 'work']
"""

min_max_scaler = MinMaxScaler()

ld = pd.read_hdf('LIVER_DATA_reduced.hdf5', key = 'df')

cols = ld.columns.drop('fail')

kb = SelectKBest(k = 160)
fail = ld['fail'].to_numpy()
ld = pd.DataFrame(kb.fit_transform(ld.drop('fail', axis = 1).to_numpy(), fail))

ld.columns = kb.get_feature_names_out(cols)

temp = ld.columns

ld = pd.DataFrame(min_max_scaler.fit_transform(ld[temp]))
ld.columns = temp

ld['fail'] = fail
train = ld.iloc[:75000]
temp = ld.iloc[75000:]
val = temp.iloc[:15000]
test = temp.iloc[15000:]

print(train['fail'].value_counts())
print(val['fail'].value_counts())

pos_train = ld.loc[ld['fail'] == 1]
neg_train = ld.loc[ld['fail'] == 0]
neg_train_balanced = neg_train.sample(n = len(pos_train.index))

balanced_train = pd.concat([pos_train, neg_train_balanced])

y_train = balanced_train['fail'].to_numpy()
x_train = balanced_train.drop('fail', axis = 1).to_numpy()

y_val = val['fail'].to_numpy()
x_val = val.drop('fail', axis = 1).to_numpy()

y_test = test['fail'].to_numpy()
x_test = test.drop('fail', axis = 1).to_numpy()


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



cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='seizure_model4.ckpt', 
                                                 monitor = 'val_auc', 
                                                 mode = 'max',
                                                 save_best_only=True,
                                                 verbose=1)

model = define_model()

history = model.fit(
        x_train,y_train,
        epochs = 400,
        validation_data = (x_val, y_val),
        callbacks=[cp_callback])

score = model.evaluate(x_test,y_test)

print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

