# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:08:44 2016

Validation of Pain Machine Learning

@author: mariarosa
"""

# Import packages
import numpy as np
import load_pain_data as pd
from scipy import stats
from sklearn import ensemble
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model


############################
# LOOCV UK and JP data
############################

X_UK = preprocessing.scale(pd.X_UK)
X_JP = preprocessing.scale(pd.X_JP)
X_US = pd.X_US
X_US = np.delete(X_US, (17), axis = 0) # Delete subject with NaNs
X_US = preprocessing.scale(X_US)
Y_US = pd.Y_US
Y_US = np.delete(Y_US, (17), axis = 0) # Delete subject with NaNs
XAll = np.vstack((X_UK,X_JP))
YAll = np.hstack((pd.Y_UK,pd.Y_JP))

XAll = preprocessing.scale(XAll)

nsite1 = pd.X_UK.shape[0]
nsite2 = pd.X_JP.shape[0]
nAll = nsite1 + nsite2
nfeatures = pd.X_UK.shape[1]

#T-test 
tt_ind = stats.ttest_ind(XAll[YAll==1,], XAll[YAll==0,])[1]<0.05   
nfeat = sum(tt_ind)

clf = ensemble.RandomForestClassifier(n_estimators=500)
#clf = svm.SVC(kernel="linear", C=0.01)
     
#Fit model  
clf.fit(XAll[::,tt_ind],YAll)
    
# Accuracy
print "Validation results:"
print clf.score(X_US[::,tt_ind],Y_US)

# Get probabilities
pb_All = clf.predict_proba(XAll[::,tt_ind])
pb_US = clf.predict_proba(X_US[::,tt_ind])

clf_lr = linear_model.LogisticRegression()
#clf_lr = svm.SVC(kernel='linear',C=1)
clf_lr.fit(pb_All, YAll)

print "Validation results (Logistic Regression):"
print clf_lr.score(pb_US, Y_US)