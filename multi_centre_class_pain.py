# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 08:45:48 2016

@author: mariarosa
"""

# Import packages
import numpy as np
import load_pain_data as pd
from scipy import stats
from sklearn import svm
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle

# Normalize features
X_UK = pd.X_UK
Y_UK = pd.Y_UK
X_JP = pd.X_JP
Y_JP = pd.Y_JP
X_US = pd.X_US
Y_US = pd.Y_US

# Classifier
clf = svm.SVC(kernel='linear')
parameters = {'C':[0.0001, 0.001, 0.01, 0.1, 1.0]}
clfgrid = GridSearchCV(clf, parameters)
clfgrid.fit(X_UK,Y_UK)

# T-test threshold
ttest_thres = 0.05

acc_UK_loo = []
acc_JP_loo = []
acc_UK_JP = []
acc_JP_UK = []
acc_UK_US = []
acc_JP_US = []
acc_All_loo = []
acc_All_US = []

############################
# Run permutations
############################

nperms = 102
for perms in range(1,nperms):
    
    print perms
    
    if (perms != 1):
        Y_UK = shuffle(Y_UK)
        Y_JP = shuffle(Y_JP)
        Y_US = shuffle(Y_US)
           
    # LOOCV UK
    print 'LOOCV-UK'
    looUK = cross_validation.LeaveOneOut(X_UK.shape[0])
    sc = []
    for train, test in looUK:
        X_train, X_test = X_UK[train,], X_UK[test,]
        Y_train, Y_test = Y_UK[train,], Y_UK[test,] 
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        tt_ind = stats.ttest_ind(X_train[Y_train==1,], X_train[Y_train==0,])[1]<ttest_thres    
        clfgrid.fit(X_train[::,tt_ind],Y_train)
        clf = svm.SVC(kernel='linear', C=clfgrid.best_params_['C'])
        clf.fit(X_train[::,tt_ind],Y_train)
        sc.append(clf.score(X_test[::,tt_ind],Y_test))
    acc_UK_loo.append(np.asarray(sc).mean())
        
    # LOOCV JP
    print 'LOOCV-JP'
    looJP = cross_validation.LeaveOneOut(X_JP.shape[0])
    sc = []
    for train, test in looJP:
        X_train, X_test = X_JP[train,], X_JP[test,]
        Y_train, Y_test = Y_JP[train,], Y_JP[test,] 
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        tt_ind = stats.ttest_ind(X_train[Y_train==1,], X_train[Y_train==0,])[1]<ttest_thres 
        clfgrid.fit(X_train[::,tt_ind],Y_train)
        clf = svm.SVC(kernel='linear', C=clfgrid.best_params_['C'])
        clf.fit(X_train[::,tt_ind],Y_train)
        sc.append(clf.score(X_test[::,tt_ind],Y_test))
    acc_JP_loo.append(np.asarray(sc).mean())
    
    # Train on UK and test on JP
    print 'UK-JP'
    X_test = X_JP
    X_test = preprocessing.scale(X_test)
    Y_test = Y_JP
    X_train = X_UK
    X_train = preprocessing.scale(X_train)
    Y_train = Y_UK
    
    tt_ind = stats.ttest_ind(X_train[Y_train==1,], X_train[Y_train==0,])[1]<ttest_thres
    clfgrid.fit(X_train[::,tt_ind],Y_train)
    clf = svm.SVC(kernel='linear', C=clfgrid.best_params_['C'])
    clf.fit(X_train[::,tt_ind],Y_train)
    coef1 = preprocessing.normalize(clf.coef_)
    int1 = preprocessing.normalize(clf.intercept_)
    acc_UK_JP.append(clf.score(X_test[::,tt_ind],Y_test))
    
    # Train on JP and test on UK
    print 'JP-UK'
    X_test = X_UK
    X_test = preprocessing.scale(X_test)
    Y_test = Y_UK
    X_train = X_JP
    X_train = preprocessing.scale(X_train)
    Y_train = Y_JP
    
    tt_ind = stats.ttest_ind(X_train[Y_train==1,], X_train[Y_train==0,])[1]<ttest_thres 
    clfgrid.fit(X_train[::,tt_ind],Y_train)
    clf = svm.SVC(kernel='linear', C=clfgrid.best_params_['C'])
    clf.fit(X_train[::,tt_ind],Y_train)
    coef2 = preprocessing.normalize(clf.coef_)
    int2 = preprocessing.normalize(clf.intercept_)
    acc_JP_UK.append(clf.score(X_test[::,tt_ind],Y_test))
    
    # Train on UK and test on US
    print 'UK-US'
    X_test = X_US
    X_test = preprocessing.scale(X_test)
    Y_test = Y_US
    X_train = X_UK
    X_train = preprocessing.scale(X_train)
    Y_train = Y_UK
    
    tt_ind = stats.ttest_ind(X_train[Y_train==1,], X_train[Y_train==0,])[1]<ttest_thres
    clfgrid.fit(X_train[::,tt_ind],Y_train)
    clf = svm.SVC(kernel='linear', C=clfgrid.best_params_['C'])  
    clf.fit(X_train[::,tt_ind],Y_train)
    acc_UK_US.append(clf.score(X_test[::,tt_ind],Y_test))
    
    # Train on JP and test on US
    print 'JP-US'
    X_test = X_US
    X_test = preprocessing.scale(X_test)
    Y_test = Y_US
    X_train = X_JP
    X_train = preprocessing.scale(X_train)
    Y_train = Y_JP
    
    tt_ind = stats.ttest_ind(X_train[Y_train==1,], X_train[Y_train==0,])[1]<ttest_thres
    clfgrid.fit(X_train[::,tt_ind],Y_train)
    clf = svm.SVC(kernel='linear', C=clfgrid.best_params_['C'])  
    clf.fit(X_train[::,tt_ind],Y_train)
    acc_JP_US.append(clf.score(X_test[::,tt_ind],Y_test))
    
    # LOOCV All
    print 'LOO-ALL'
    looAll = cross_validation.LeaveOneOut(X_JP.shape[0]+X_UK.shape[0])
    
    X_All = np.vstack((X_UK,X_JP))
    Y_All = np.hstack((Y_UK,Y_JP))
    
    sc = []
    for train, test in looAll:
        X_train, X_test = X_All[train,], X_All[test,]
        Y_train, Y_test = Y_All[train,], Y_All[test,]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        tt_ind = stats.ttest_ind(X_train[Y_train==1,], X_train[Y_train==0,])[1]<ttest_thres
        clfgrid.fit(X_train[::,tt_ind],Y_train)
        clf = svm.SVC(kernel='linear', C=clfgrid.best_params_['C'])
        clf.fit(X_train[::,tt_ind],Y_train)
        sc.append(clf.score(X_test[::,tt_ind],Y_test))
    acc_All_loo.append(np.asarray(sc).mean())
    
    # Train All test US
    print 'ALL-US'
    X_test = X_US
    X_test = preprocessing.scale(X_test)
    Y_test = Y_US
    X_train = X_All
    X_train = preprocessing.scale(X_train)
    Y_train = Y_All
    
    tt_ind = stats.ttest_ind(X_train[Y_train==1,], X_train[Y_train==0,])[1]<ttest_thres
    clfgrid.fit(X_train[::,tt_ind],Y_train)
    clf = svm.SVC(kernel='linear', C=clfgrid.best_params_['C'])   
    clf.fit(X_train[::,tt_ind],Y_train)
    acc_All_US.append(clf.score(X_test[::,tt_ind],Y_test))
  
print "P-values:" 
if (nperms-1) == 1:
    print acc_UK_loo[0]
    print acc_JP_loo[0]
    print acc_UK_JP[0]
    print acc_JP_UK[0]
    print acc_UK_US[0]
    print acc_JP_US[0]
    print acc_All_loo[0]
    print acc_All_US[0]
    
if nperms > 2: 
    print sum((acc_UK_loo[1::] >= acc_UK_loo[0]))/float(perms-1)
    print sum(acc_JP_loo[1::] >= acc_JP_loo[0])/float(perms-1)
    print sum(acc_UK_JP[1::] >= acc_UK_JP[0])/float(perms-1)
    print sum(acc_JP_UK[1::] >= acc_JP_UK[0])/float(perms-1)
    print sum(acc_UK_US[1::] >= acc_UK_US[0])/float(perms-1)
    print sum(acc_JP_US[1::] >= acc_JP_US[0])/float(perms-1)
    print sum(acc_All_loo[1::] >= acc_All_loo[0])/float(perms-1)
    print sum(acc_All_US[1::] >= acc_All_US[0])/float(perms-1)