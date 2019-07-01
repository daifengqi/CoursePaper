# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:11:12 2019

@author: Administrator
"""
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.metrics import classification_report

from sklearn import tree

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

'''
test = pd.concat([X,y],axis=1)
fig, axes = plt.subplots(1,2)
test.boxplot(column = 'dct_pdays',by = 'y',ax=axes[0])
test.boxplot(column = 'dct_previous',by = 'y',ax=axes[1])
fig.suptitle('')
'''

seed = 1
# LR
clf_l1_LR = LogisticRegression(penalty='l1', solver='saga',random_state = seed, max_iter=100)
clf_svm = SVC(random_state = seed, probability=True)
clf_tree = tree.DecisionTreeClassifier()
clf_RF = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0,bootstrap=True,criterion='gini')
clf_GBDT = ensemble.GradientBoostingClassifier(loss='exponential', max_depth= 5, max_features= None, n_estimators= 100, subsample= 0.7,random_state = seed)

def roc_plot(X, y, classifier, cls_name, colorin):
    cv = StratifiedKFold(n_splits=6)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train,:], y[train]).predict_proba(X.iloc[test,:])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    
        i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #          label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=colorin,
             label=cls_name + r' (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('')
    plt.legend(loc="lower right")
    # plt.show()
    
roc_plot(X,y,clf_l1_LR, 'Logistic Regression','blue')
roc_plot(X,y,clf_svm, 'SVM', 'aquamarine')
roc_plot(X,y,clf_tree, 'Decision Tree','lightblue')
roc_plot(X,y,clf_RF, 'Random Forest','darkorange')
roc_plot(X,y,clf_GBDT, 'GBDT','green')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
plt.show()

# Tree-based selectino
rf_select = RandomForestClassifier(criterion='gini', random_state=1).fit(X, y)
importances = rf_select.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_select.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
feat_labels = X.columns[indices]
# Print the feature ranking
print("Feature ranking:")

# for f in range(X.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))


X = X.loc[:,feat_labels[0:21]]
clf_LR = LogisticRegression()
clf_svm = SVC(random_state = seed,probability=True)
clf_tree = tree.DecisionTreeClassifier()
clf_RF = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0,bootstrap=True,criterion='gini')
clf_GBDT = ensemble.GradientBoostingClassifier(loss='exponential', max_depth= 5, max_features= None, n_estimators= 100, subsample= 0.7,random_state = seed)
roc_plot(X,y,clf_LR, 'Logistic Regression','blue')
roc_plot(X,y,clf_svm, 'SVM','aquamarine')
roc_plot(X,y,clf_tree, 'Decision Tree','lightblue')
roc_plot(X,y,clf_RF, 'Random Forest','darkorange')
roc_plot(X,y,clf_GBDT, 'GBDT','green')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
plt.show()

