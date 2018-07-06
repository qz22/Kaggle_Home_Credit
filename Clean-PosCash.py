# With mean encoding and extra features
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
from sklearn import *
import re
from catboost import CatBoostClassifier
from sklearn.model_selection import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from matplotlib import pyplot
import seaborn as sns
'''
DATA_FOLDER = '/Users/Sonia/Desktop/Kaggle/HomeCredit'
os.chdir(DATA_FOLDER)
poscash = pd.read_csv('POS_CASH_balance.csv')
'''
mean_cols = poscash.NAME_CONTRACT_STATUS.unique()
poscash = pd.concat( [poscash , pd.get_dummies( poscash[ 'NAME_CONTRACT_STATUS' ])], axis=1 )
poscash.drop([ 'NAME_CONTRACT_STATUS' ], axis = 1, inplace = True )
grp_poscash = poscash.groupby( ['SK_ID_CURR'] ).SK_ID_PREV.nunique().reset_index()
poscash[ 'CNT_INSTALMENT_FUTURE' ] = poscash[ 'CNT_INSTALMENT_FUTURE' ] / (poscash[ 'CNT_INSTALMENT' ] + 1.0 )
cols = poscash.columns.values
for col in cols:
    if not col == 'SK_ID_CURR' and not col == 'SK_ID_PREV':
        if not poscash[ col ].dtypes == "object":
            if col == 'MONTHS_BALANCE':
                f = { col: [ 'min', 'max', 'median']}
            elif col in mean_cols:
                f = { col: [ 'mean' ]}
            else:
                f = { col :['median', 'min', 'max', 'var']}
            grp1 = poscash.groupby( 'SK_ID_CURR' ).agg(f).reset_index()
            grp_poscash = grp_poscash.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
grp_poscash.to_csv( 'Clean-PosCash.csv', index = False )


# clean data
credit.isnull().sum()

'''
    SK_ID_PREV                         0
    SK_ID_CURR                         0
    MONTHS_BALANCE                     0
    AMT_BALANCE                        0
    AMT_CREDIT_LIMIT_ACTUAL            0
    AMT_DRAWINGS_ATM_CURRENT      749816
    AMT_DRAWINGS_CURRENT               0
    AMT_DRAWINGS_OTHER_CURRENT    749816
    AMT_DRAWINGS_POS_CURRENT      749816
    AMT_INST_MIN_REGULARITY       305236
    AMT_PAYMENT_CURRENT           767988
    AMT_PAYMENT_TOTAL_CURRENT          0
    AMT_RECEIVABLE_PRINCIPAL           0
    AMT_RECIVABLE                      0
    AMT_TOTAL_RECEIVABLE               0
    CNT_DRAWINGS_ATM_CURRENT      749816
    CNT_DRAWINGS_CURRENT               0
    CNT_DRAWINGS_OTHER_CURRENT    749816
    CNT_DRAWINGS_POS_CURRENT      749816
    CNT_INSTALMENT_MATURE_CUM     305236
    NAME_CONTRACT_STATUS               0
    SK_DPD                             0
    SK_DPD_DEF                         0
'''