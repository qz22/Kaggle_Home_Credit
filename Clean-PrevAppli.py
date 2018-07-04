# -*- coding: utf-8 -*-
# Preprocess 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import re
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
import seaborn as sns
'''
DATA_FOLDER = '/Users/Sonia/Desktop/Kaggle/HomeCredit'
os.chdir(DATA_FOLDER)
prev = pd.read_csv('previous_application.csv') # (1670214, 37)
'''

# Usually used functions
import pandas as pd
# CateToPercent: given a categorical column(col), and a groupby Key (key), calulate the 
# percentage of different categories in the sample for each Key
# For example, given the history records of loan applications, given Status (col), what the 
# percentage of refused (or approved) apllication for each Id (key)
def CateToPercent( df, col, key, obj, sum_col, pct ):
    temp = df.loc[ df[col].isnull() != True ]
    types = temp[ col ].unique()
    df = pd.concat( [ df, pd.get_dummies( df[ col ])],axis=1 ).copy()
    for t in types[ : -1 ]: # to avoid multicollinearity
        grp = df.groupby( [ key ] )[ t ].sum().reset_index()
        obj =  obj.merge( grp, on = [ key ], how = 'left')
        if pct:
            obj[ t + '_pct'] = ( obj[ t ] / ( obj[ sum_col ] + 1 ) ).copy()
            obj.drop( [ t ], axis = 1, inplace = True)
    return( obj )

def GroupByMEAN( df, obj , col, key ):
    temp = df.loc[ df[ col].isnull() == False ].copy()
    grp = temp.groupby( [ key ] )[col].mean().reset_index()
    obj =  obj.merge(grp, on = [ key ], how = 'left')
    return( obj )

# Calculate the ratio of two columns (col1, col2 )
def RatioOfTwo( df, obj, col1, col2, key ):
    temp = df.loc[ df[ col1 ].isnull() == False ].copy()
    grp = temp.groupby( [ key ] )[ col1 ].sum().reset_index()
    obj =  obj.merge(grp, on = [ key ], how = 'left').fillna( grp[ col1 ].median())
    grp = temp.groupby( [ key ] )[ col2 ].sum().reset_index()
    obj =  obj.merge( grp, on = [ key ], how = 'left').fillna( grp[ col2 ].median())
    obj[ col1 + '_Pct' ] = obj[ col1 ] / ( obj[ col2 ] + 1 )
    obj.drop( [col1, col2 ], axis = 1, inplace = True )
    return( obj )

# Fill null values of AMT_CREDIT
prev.loc[ (prev.AMT_APPLICATION == 0 ) & (prev.AMT_CREDIT!= 0 ), 'AMT_APPLICATION'] = \
prev.loc[ (prev.AMT_APPLICATION == 0 ) & (prev.AMT_CREDIT!= 0 ), 'AMT_CREDIT']
prev = prev.loc[prev.AMT_CREDIT.isnull() != True ]
grp_prev = prev.groupby( [ 'SK_ID_CURR' ] ).SK_ID_PREV.count().reset_index().rename(index=str, columns={'SK_ID_PREV': 'PREV_APPLI_COUNT'})
for col in prev.columns.values:
    print( col )
    if col == 'SK_ID_CURR' or col == 'SK_ID_PREV':
        continue
    if prev[ col ].dtypes == "object":
        print( len( prev[ col ].unique()))
        if len( prev[ col ].unique())==2:
            le = LabelEncoder()
            le.fit( prev[ col ])
            prev[ col ] = le.transform( prev[ col ] )
            grp_prev = GroupByMEAN( prev,  grp_prev , col, 'SK_ID_CURR' )
        if len( prev[ col ].unique()) > 2:
            grp_prev = CateToPercent( prev, col, 'SK_ID_CURR', grp_prev, 'PREV_APPLI_COUNT', 0 );
    else:
        grp_prev = GroupByMEAN( prev,  grp_prev , col, 'SK_ID_CURR' )

grp_prev.to_csv('Clean-PrevAppli.csv', index = False)


grp[ 'Refused' ] = grp[ 'Refused' ] > 0
    grp = RatioOfTwo( prev, grp, 'AMT_APPLICATION' , 'AMT_CREDIT', key )
    prev['DAYS_DECISION'] = prev['DAYS_DECISION']/365 * -1
    # 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'RATE_DOWN_PAYMENT', too much noise
    for col in [ 'DAYS_DECISION' ]:
        grp = GroupByMEAN( prev, grp, col, key )
    # , 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE'
    for col in [ 'NAME_PAYMENT_TYPE', 'NAME_CLIENT_TYPE' ]:
        grp = CateToPercent( prev, col, 'SK_ID_CURR', grp, 'PREV_APPLI_COUNT', 1 )
    return( grp )
'''
SK_ID_PREV                           0
SK_ID_CURR                           0
NAME_CONTRACT_TYPE                   0
AMT_ANNUITY                     372235
AMT_APPLICATION                      0
AMT_CREDIT                           1
AMT_DOWN_PAYMENT                895844
AMT_GOODS_PRICE                 385515
WEEKDAY_APPR_PROCESS_START           0
HOUR_APPR_PROCESS_START              0
FLAG_LAST_APPL_PER_CONTRACT          0
NFLAG_LAST_APPL_IN_DAY               0
RATE_DOWN_PAYMENT               895844
RATE_INTEREST_PRIMARY          1664263
RATE_INTEREST_PRIVILEGED       1664263
NAME_CASH_LOAN_PURPOSE               0
NAME_CONTRACT_STATUS                 0
DAYS_DECISION                        0
NAME_PAYMENT_TYPE                    0
CODE_REJECT_REASON                   0
NAME_TYPE_SUITE                 820405
NAME_CLIENT_TYPE                     0
NAME_GOODS_CATEGORY                  0
NAME_PORTFOLIO                       0
NAME_PRODUCT_TYPE                    0
CHANNEL_TYPE                         0
SELLERPLACE_AREA                     0
NAME_SELLER_INDUSTRY                 0
CNT_PAYMENT                     372230
NAME_YIELD_GROUP                     0
PRODUCT_COMBINATION                346
DAYS_FIRST_DRAWING              673065
DAYS_FIRST_DUE                  673065
DAYS_LAST_DUE_1ST_VERSION       673065
DAYS_LAST_DUE                   673065
DAYS_TERMINATION                673065
NFLAG_INSURED_ON_APPROVAL       673065
'''