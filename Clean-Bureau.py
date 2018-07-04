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
bureau = pd.read_csv('bureau.csv') # (1716428, 17)
bb = pd.read_csv( 'bureau_balance.csv' )
'''
def CateToPercent( df, col, key, obj, sum_col, pct ):
    """
    CateToPercent: given a categorical column(col), and a groupby Key (key), calulate the
    percentage of different categories in the sample for each Key
    For example, given the history records of loan applications, given Status (col), what the
    percentage of refused (or approved) apllication for each Id (key)"""
    types = df[ col ].unique()
    df = pd.concat( [ df, pd.get_dummies( df[ col ])],axis=1 ).copy()
    for t in types[ : -1 ]: # to avoid multicollinearity
        grp = df.groupby( [ key ] )[ t ].sum().reset_index()
        obj =  obj.merge( grp, on = [ key ], how = 'left')
        if pct:
            obj[ t + '_pct'] = ( obj[ t ] / ( obj[ sum_col ] + 1 ) ).copy()
            obj.drop( [ t ], axis = 1, inplace = True)
    return( obj )

# clean data
bureau[ 'DAYS_CREDIT'] = bureau.DAYS_CREDIT /365 * (-1)
bureau[ 'CREDIT_DAY_OVERDUE'] = bureau.CREDIT_DAY_OVERDUE /365
bureau[ 'OVERDUE']= bureau[ 'CREDIT_DAY_OVERDUE'] == 0
#fill null values of AMT_CREDIT_SUM
bureau.loc[(bureau.AMT_CREDIT_SUM.isnull() == 1 ), 'AMT_CREDIT_SUM' ] = bureau.loc[(bureau.AMT_CREDIT_SUM.isnull() == 1 ), 'AMT_CREDIT_SUM_DEBT' ]
# Clean AMT_CREDIT_SUM_DEBT
bureau.loc[( ( bureau.AMT_CREDIT_SUM_DEBT.isnull() == 1 ) & (bureau.CREDIT_ACTIVE == 'Closed') ), 'AMT_CREDIT_SUM_DEBT' ] = 0
bureau.loc[( ( bureau.AMT_CREDIT_SUM_DEBT.isnull() == 1 ) & (bureau.CREDIT_ACTIVE != 'Closed') & (bureau.AMT_CREDIT_SUM_LIMIT.isnull() == False) ), 'AMT_CREDIT_SUM_DEBT' ] = \
bureau.loc[( ( bureau.AMT_CREDIT_SUM_DEBT.isnull() == 1 ) & (bureau.CREDIT_ACTIVE != 'Closed') & (bureau.AMT_CREDIT_SUM_LIMIT.isnull() == False) ), 'AMT_CREDIT_SUM' ] - \
bureau.loc[( ( bureau.AMT_CREDIT_SUM_DEBT.isnull() == 1 ) & (bureau.CREDIT_ACTIVE != 'Closed') & (bureau.AMT_CREDIT_SUM_LIMIT.isnull() == False ) ), 'AMT_CREDIT_SUM_LIMIT' ]
bureau.loc[( bureau.AMT_CREDIT_SUM_DEBT.isnull() == 1 ), 'AMT_CREDIT_SUM_DEBT' ] = \
bureau.loc[( bureau.AMT_CREDIT_SUM_DEBT.isnull() == 1 ), 'AMT_CREDIT_SUM' ]
# Number of loans
grp = bureau[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})
# Percentage of OverDue
grp1 = bureau[['SK_ID_CURR', 'OVERDUE']].groupby(by = ['SK_ID_CURR'])['OVERDUE'].mean().reset_index()
grp = grp.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
# Sum of Debt/credit
grp1 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index()
grp = grp.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
grp1 = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index()
grp = grp.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
# domain knowledge
grp['DEBT_CREDIT'] = grp.AMT_CREDIT_SUM_DEBT / (grp.AMT_CREDIT_SUM+1)
grp = CateToPercent( bureau, 'CREDIT_ACTIVE', 'SK_ID_CURR', grp, 'BUREAU_LOAN_COUNT', 1 )
grp = CateToPercent( bureau, 'CREDIT_CURRENCY', 'SK_ID_CURR', grp, 'BUREAU_LOAN_COUNT', 1 )
grp = CateToPercent( bureau, 'CREDIT_TYPE', 'SK_ID_CURR', grp, 'BUREAU_LOAN_COUNT', 1 )

grp1 = bureau[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT'].mean().reset_index()
grp = grp.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
grp.to_csv("Clean-Bureau.csv", index=Falsedata = pd.read_csv( 'Clean-appli.csv' )

'''
Is Null:
SK_ID_CURR                      0
SK_ID_BUREAU                    0
CREDIT_ACTIVE                   0
CREDIT_CURRENCY                 0
DAYS_CREDIT                     0
CREDIT_DAY_OVERDUE              0
DAYS_CREDIT_ENDDATE        105553
DAYS_ENDDATE_FACT          633653
AMT_CREDIT_MAX_OVERDUE    1124488
CNT_CREDIT_PROLONG              0
AMT_CREDIT_SUM                 13
AMT_CREDIT_SUM_DEBT        257669
AMT_CREDIT_SUM_LIMIT       591780
AMT_CREDIT_SUM_OVERDUE          0
CREDIT_TYPE                     0
DAYS_CREDIT_UPDATE              0
AMT_ANNUITY               1226791

'''