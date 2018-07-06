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
install0 = pd.read_csv('installments_payments.csv')
'''
grp_install = install.groupby( ['SK_ID_CURR'] ).SK_ID_PREV.nunique().reset_index()
install[ 'AMT_PAYMENT' ] = install[ 'AMT_PAYMENT' ] / (install[ 'AMT_INSTALMENT' ] + 1.0 )
install[ 'DAYS_PAY' ] = (install[ 'DAYS_INSTALMENT' ] - install[ 'DAYS_ENTRY_PAYMENT' ]) / ( install[ 'DAYS_ENTRY_PAYMENT' ] + 1.0 )
install.drop( [ 'DAYS_ENTRY_PAYMENT', 'DAYS_INSTALMENT', 'AMT_INSTALMENT' ], axis = 1, inplace = True )

cols = install.columns.values
for col in cols:
    if not col == 'SK_ID_CURR' and not col == 'SK_ID_PREV' :
        if not install[ col ].dtypes == "object":
            f = { col: [ 'min', 'max', 'median', 'var']}
            grp1 = install.groupby( 'SK_ID_CURR' ).agg(f).reset_index()
            grp_install = grp_install.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
grp_install.to_csv( 'Clean-Installment.csv', index = False)

