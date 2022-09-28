import pandas as pd
import numpy as np
import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 


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
        #(ld.PX_STAT != "") &
        (pd.isna(ld.LIV_DON_TY))]
        #(ld.INIT_MELD_PELD_LAB_SCORE < 15)

ld['ABO_A'] = np.where(ld['ABO'] == 'A', 1, 0)
ld['ABO_B'] = np.where(ld['ABO'] == 'B', 1, 0)
ld['ABO_AB'] = np.where(ld['ABO'] == 'AB', 1, 0)
ld['ABO_O'] = np.where(ld['ABO'] == 'O', 1, 0)
ld['ABO_MAT'] = np.where(ld['ABO_MAT'] == 0, 0, np.where(ld['ABO_MAT'] == 1, 1, 2))
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

ld['fail'] = np.where((ld['DEATH_DATE'].isnull() == False) & (ld['TX_DATE'].isnull() == True), 1, 0)

print(ld['fail'].value_counts())


values = ['PORTAL_VEIN_TCR', 'PORTAL_VEIN_TRR',	'PREV_AB_SURG_TCR',	'PREV_AB_SURG_TRR',	'PREV_TX', 'MALIG_TCR', 'NON_HRT_DON', 'MALIG', 'LIFE_SUP_TRR', 'LIFE_SUP_TCR', 
          'INSULIN_DON', 'INOTROP_SUPPORT_DON', 'INIT_DIALYSIS_PRIOR_WEEK', 'HISTORY_MI_DON', 'HIST_OTH_DRUG_DON', 'HIST_INSULIN_DEP_DON', 'HIST_CIG_DON', 'HIST_CANCER_DON', 'HEPARIN_DON',
          'DIAL_TX', 'DIABETES_DON', 'DDAVP_DON', 'CLIN_INFECT_DON', 'CARDARREST_NEURO', 'BACT_PERIT_TCR', 'ANTIHYPE_DON', 'PROTEIN_URINE',	'PT_DIURETICS_DON',	'PT_OTH_DON',	'PT_STEROIDS_DON',	'PT_T3_DON',
           'PT_T4_DON',	'RECOV_OUT_US', 'TATTOOS', 'TIPSS_TCR',	'TIPSS_TRR','VASODIL_DON','VDRL_DON', 'WORK_INCOME_TCR', 'WORK_INCOME_TRR','HIST_COCAINE_DON', 'CDC_RISK_HIV_DON', 'ARGININE_DON']

for value in values:
    ld[value] = np.where(ld[value] == 'Y', 1, 0)

features = pd.read_csv('waitlist_features.csv')

feature_list = features.values.tolist()

ld_reduced = pd.DataFrame()

for feature in feature_list:
    ld_reduced[feature[0]] = ld[feature[0]]

cols = ld_reduced.columns[ld_reduced.isnull().mean()>0.25]
ld_reduced.drop(cols, axis=1)

cols = ld_reduced.columns


imp = IterativeImputer(max_iter = 20)
ld_reduced = pd.DataFrame(imp.fit_transform(ld_reduced))
ld_reduced.columns = cols

ld_reduced.to_hdf('LIVER_DATA_allwaitlist.hdf5', key = 'df')

