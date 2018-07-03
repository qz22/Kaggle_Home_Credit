# OHE + generating additional features
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
from sklearn import *
import re
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
import seaborn as sns

DATA_FOLDER = '/Users/Sonia/Desktop/Kaggle/HomeCredit'
os.chdir(DATA_FOLDER)
train    = pd.read_csv('application_train.csv') #(891, 12)
test    = pd.read_csv('application_test.csv') #(891, 12)

train.select_dtypes( include = ['object']).apply(pd.Series.nunique, axis = 0)

for col in train.columns.values:
    if train[ col ].dtypes == "object":
        if len( train[ col ].unique())==2:
            le = LabelEncoder()
            le.fit( train[ col ])
            train[ col ] = le.transform( train[ col ] )
            test[ col ] = le.transform( test[col] )
train = pd.get_dummies( train )
test = pd.get_dummies( test )
y = train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
train, test = train.align( test, join = 'inner', axis = 1)
train['TARGET'] = y

data = pd.concat([train_x,test],axis=0)
ANOM_DAYS_EMPLOYED = max( data.DAYS_EMPLOYED)
data['DAYS_EMPLOYED_ANOM'] = data["DAYS_EMPLOYED"] == ANOM_DAYS_EMPLOYED
data['DAYS_EMPLOYED'].replace({ ANOM_DAYS_EMPLOYED: np.nan}, inplace = True)

# correlations = train.corr()['TARGET']
plt.figure(figsize = (10, 8))
sns.kdeplot( train.loc[ train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')
sns.kdeplot( train.loc[ train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages')
# Fill null values
train[ 'AMT_INCOME_CATE' ] = pd.cut( train[ 'AMT_INCOME_TOTAL' ], bins = np.linspace( 0 ,max(train[ 'AMT_INCOME_TOTAL' ]) , num = 10))
train[ 'AMT_CREDIT_CATE' ] = pd.cut( train[ 'AMT_CREDIT' ], bins = np.linspace( 0 , max(train[ 'AMT_CREDIT' ]), num = 10))
data[ 'AMT_INCOME_CATE' ] = pd.cut( data[ 'AMT_INCOME_TOTAL' ], bins = np.linspace( 0 ,max(train[ 'AMT_INCOME_TOTAL' ]) , num = 10))
data[ 'AMT_CREDIT_CATE' ] = pd.cut( data[ 'AMT_CREDIT' ], bins = np.linspace( 0 , max(train[ 'AMT_CREDIT' ]), num = 10))
## Fill null values of AMT_ANUITY with median in each group of AMT_INCOME_CATE
temp1 = train[ [ 'AMT_INCOME_TOTAL', 'AMT_INCOME_CATE', 'AMT_ANNUITY']].dropna().copy()
map1 = temp1.groupby([ 'AMT_INCOME_CATE' ]).AMT_ANNUITY.median()
data[ 'AMT_INCOME_MEDIAN' ] = data[ 'AMT_INCOME_CATE'].apply( lambda x : map1[x ])
data[['AMT_ANNUITY']] = data[['AMT_ANNUITY']].apply(lambda x: x.fillna(value= data[ 'AMT_INCOME_MEDIAN']))
data.drop( [ 'AMT_INCOME_MEDIAN', 'AMT_INCOME_CATE'], axis = 1, inplace = True )
 # Fill null values of AMT_GOODS_PRICE with median in each group of AMT_CREDIT_CATE 
temp2 = train[ [ 'AMT_CREDIT_CATE', 'AMT_GOODS_PRICE']].dropna().copy()
map2 = temp2.groupby([ 'AMT_CREDIT_CATE' ]).AMT_GOODS_PRICE.median()
data[ 'AMT_GOODS_PRICE_MEDIAN' ] = data[ 'AMT_CREDIT_CATE'].apply( lambda x : map2[x ])
data[['AMT_GOODS_PRICE']] = data[['AMT_GOODS_PRICE']].apply(lambda x: x.fillna(value= data[ 'AMT_GOODS_PRICE_MEDIAN']))
data.drop( [ 'AMT_GOODS_PRICE_MEDIAN', 'AMT_CREDIT_CATE'], axis = 1, inplace = True ) 

data[ 'DAYS_BIRTH_CATE'] = round( data['DAYS_BIRTH'] * (-1) / 365 / 10)
train[ 'DAYS_BIRTH_CATE'] = round( train['DAYS_BIRTH'] * (-1) / 365 / 10 )
temp3 = train[ [ 'DAYS_BIRTH_CATE', 'DAYS_EMPLOYED']].dropna().copy()
map3 = temp3.groupby([ 'DAYS_BIRTH_CATE' ]).DAYS_EMPLOYED.median()
data[ 'DAYS_EMPLOYED_MEDIAN' ] = data[ 'DAYS_BIRTH_CATE'].apply( lambda x : map3[x ])
data[['DAYS_EMPLOYED']] = data[['DAYS_EMPLOYED']].apply(lambda x: x.fillna(value= data[ 'DAYS_EMPLOYED_MEDIAN']))
data.drop( [ 'DAYS_EMPLOYED_MEDIAN', 'DAYS_BIRTH_CATE' ], axis = 1, inplace =True )

# Generate new features based on credit background
data[ 'ANNUITY-INCOME' ] = data[ 'AMT_ANNUITY' ]/ data['AMT_INCOME_TOTAL']
data[ 'CREDIT-INCOME' ] = data[ 'AMT_CREDIT' ]/ ( data['AMT_INCOME_TOTAL'] + data[ 'AMT_ANNUITY' ] )
data[ 'LTV' ] = data[ 'AMT_CREDIT' ] / data[ 'AMT_GOODS_PRICE' ]
data[ 'DAYS_EMPLOYED_PCT' ] = data[ 'DAYS_EMPLOYED']/data['DAYS_BIRTH']
data[ 'Child-Fam' ] = data['CNT_CHILDREN'] / (data[ 'CNT_FAM_MEMBERS' ] + 1 )

# data = pd.read_csv( 'appli_clnd.csv' )
data.drop( [ 'SK_ID_CURR'], axis = 1, inplace = True  )
from sklearn.preprocessing import MinMaxScaler, Imputer
train_x = data[:train.shape[0]]
test_x = data[train.shape[0]:]

features = list(train_x.columns)
# Median imputation of missing values
imputer = Imputer(strategy = 'median')
# Fit on the training data
imputer.fit(train_x)
# Transform both training and testing data
train_x = imputer.transform(train_x)
test_x = imputer.transform(test_x)
# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

y = train[ 'TARGET']
colnames = data.columns.values
lgb_pred, feat_imp, metrics = model( train_x, y, test_x, colnames, n_folds = 5)
sub_df = pd.DataFrame({ 'SK_ID_CURR': test['SK_ID_CURR'], 'TARGET': lgb_pred['TARGET']})

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

# Thanks to Will Koehrsen for the great analysis
def model( x, y, test_data, feature_names, n_folds = 5 ):   
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
    cat_indices = 'auto'
    # Empty array for feature importances
    feature_importance_values = np.zeros( x.shape[1] )
    # Empty array for test predictions
    test_predictions = np.zeros(test_data.shape[0])
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros( x.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split( x ):     
        train_features, train_labels = x[train_indices], y[train_indices] # Training data for the fold
        valid_features, valid_labels = x[valid_indices], y[valid_indices]# Validation data for the fold
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
         # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        # Make predictions
        test_predictions += model.predict_proba(test_data, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'TARGET': test_predictions })
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score( y, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics    

# EDA of numeric/categorical features
def inspect_feature_plot(data, feat):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))
    sns.barplot(data[feat], data.TARGET, ax=ax1)
    sns.countplot(data[feat], ax=ax2)

def inspect_continuous_feature_plot(data, feat):
    data = data[ [  feat]].dropna().copy()
    sns.distplot( data[ feat ].values)
 