# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 22:26:34 2019

@author: Administrator
"""

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

data_train = pd.read_csv('Train.csv')

X = data_train.iloc[:,0:20]
y = data_train.iloc[:,20]
y.loc[y=='yes'] = 1
y.loc[y=='no'] = 0


# Feature Engineering
x1 = X.loc[:,'pdays'].copy()
x1[x1==999] = 1
x1[x1>1] = 0
x2 = X.loc[:,'previous'].copy()
x2[x2!=0] = 1
x1 = x1.rename('dct_pdays')
x2 = x2.rename('dct_previous')


# Scaler
X_num = X.loc[:,['age','duration','campaign','pdays','previous',
                 'emp.var.rate','cons.price.idx','cons.conf.idx',
                 'euribor3m','nr.employed']]
# X_num = X_num.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# String Variables
X_str = X.loc[:,['job','marital','education','default','housing',
                 'loan','contact','month','day_of_week','poutcome']]
X_dummy = pd.get_dummies(X_str)

# Merge Datafrane
seed = 1
X = pd.concat([X_num,X_dummy,x1,x2],axis=1)
X = X.drop('default_yes',1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'gamma': 0.01,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
    'eval_metric':'auc'
}

num_round = 100
bst = xgb.train(params, dtrain, num_round)
preds = bst.predict(dtest)
fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=2)
auc_score = roc_auc_score(y_test, preds)

X_test = pd.read_csv('Test.csv')
# Feature Engineering
x1 = X_test.loc[:,'pdays'].copy()
x1[x1==999] = 1
x1[x1>1] = 0
x2 = X_test.loc[:,'previous'].copy()
x2[x2!=0] = 1
x1 = x1.rename('dct_pdays')
x2 = x2.rename('dct_previous')

# Scaler
X_num = X_test.loc[:,['age','duration','campaign','pdays','previous',
                 'emp.var.rate','cons.price.idx','cons.conf.idx',
                 'euribor3m','nr.employed']]
# X_num = X_num.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# String Variables
X_str = X_test.loc[:,['job','marital','education','default','housing',
                 'loan','contact','month','day_of_week','poutcome']]
X_dummy = pd.get_dummies(X_str)


# Merge Datafrane
seed = 1
X = pd.concat([X_num,X_dummy,x1,x2],axis=1)
dtest = xgb.DMatrix(X)

predictY = bst.predict(dtest)
predictY = pd.DataFrame(predictY)
print(np.sum(predictY)/10000)
predictY.to_csv('Results_1.csv', encoding = 'utf-8', index=False , header=False)
