# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:42:15 2019

@author: Cesc
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.metrics import classification_report
import datetime
from sklearn import tree

starttime = datetime.datetime.now()
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)


data_train = pd.read_csv('Train.csv')
# print(data_train.describe())
# Y~ age
'''
fig, axes = plt.subplots(1,2)
data_train.boxplot(column = 'age',by = 'y',ax=axes[0])
data_train.boxplot(column = 'duration',by = 'y',ax=axes[1])
axes[0].set_xlabel('Contract Extension')
axes[0].set_ylabel('Distribution of Numeric Variables')
axes[1].set_xlabel('Contract Extension')
axes[1].set_ylabel(' ')
fig.suptitle('')
fig.show()
'''

X = data_train.iloc[:,0:20]
y = data_train.iloc[:,20]
y.loc[y=='yes'] = 1
y.loc[y=='no'] = 0

# image
'''
fig, axes = plt.subplots()
plt.bar(x = ['yes','no'], height = [np.sum(y), data_train.shape[0]-np.sum(y)],
        align = 'center', color = ['blue','lightblue'])
plt.title('')
plt.xlabel('Contract Extension')
plt.ylabel('Number of People')
plt.show()
'''

# EDA
# fig, axes = plt.subplots()


# print(np.sum(y))
# print(np.sum(pd.isna(X))) without missing values

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
X_num = X_num.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# String Variables
X_str = X.loc[:,['job','marital','education','default','housing',
                 'loan','contact','month','day_of_week','poutcome']]
X_dummy = pd.get_dummies(X_str)

# Merge Datafrane
X = pd.concat([X_num,X_dummy,x1,x2],axis=1)
X = X.drop('default_yes',1)

'''
test = pd.concat([X,y],axis=1)
fig, axes = plt.subplots(1,2)
test.boxplot(column = 'dct_pdays',by = 'y',ax=axes[0])
test.boxplot(column = 'dct_previous',by = 'y',ax=axes[1])
fig.suptitle('')
'''

seed = 1
# LR penalty='l1', solver='saga',random_state = seed, max_iter=100
clf_LR = LogisticRegression()
s1 = datetime.datetime.now()
scores_lr = cross_val_score(clf_LR, X, y, cv=5, scoring='roc_auc')
s2 = datetime.datetime.now()
print((s2-s1))
print("LR-AUC: %0.2f (+/- %0.2f)" % (scores_lr.mean(), scores_lr.std() * 2))

# SVM
clf_svm = SVC(random_state = seed)
s1 = datetime.datetime.now()
scores = cross_val_score(clf_svm, X, y, cv=5, scoring='roc_auc')
s2 = datetime.datetime.now()
print((s2-s1))
print("SVM-AUC: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

# DT
clf_tree = tree.DecisionTreeClassifier()
s1 = datetime.datetime.now()
scores = cross_val_score(clf_tree, X, y, cv=5, scoring='roc_auc')
s2 = datetime.datetime.now()
print((s2-s1))
print("DT-AUC: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

# RF
clf_RF = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0,bootstrap=True,criterion='gini')
s1 = datetime.datetime.now()
scores = cross_val_score(clf_RF, X, y, cv=5, scoring='roc_auc')
s2 = datetime.datetime.now()
print((s2-s1))
print("RF-AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# GBDT
clf_GBDT = ensemble.GradientBoostingClassifier(loss='exponential', max_depth= 5, max_features= None, n_estimators= 100, subsample= 0.7,random_state = seed)
s1 = datetime.datetime.now()
scores = cross_val_score(clf_GBDT, X, y, cv=5, scoring='roc_auc')
s2 = datetime.datetime.now()
print((s2-s1))
print("GBDT-AUC: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))





