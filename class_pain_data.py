# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:08:37 2016

@author: mariarosa
"""

# Import packages
import numpy as np
import load_pain_data as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import svm
from sklearn import ensemble
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import decomposition
from sklearn.cross_validation import KFold

# Normalize features
#X_UK = preprocessing.scale(pd.X_UK)
#Y_UK = pd.Y_UK
#X_JP = preprocessing.scale(pd.X_JP)
#Y_JP = pd.Y_JP

# Train on JP and test on JP
#Xtest = X_JP
#Ytest = Y_JP
#Xtrain = X_UK
#Ytrain = Y_UK
#

############################
# Run standard classifiers
############################

#clf = svm.SVC()
#clf.fit(Xtrain,Ytrain)

#print "SVM accuracy"
#print clf.score(Xtest,Ytest)

#clf = ensemble.RandomForestClassifier(n_estimators=500, max_features = nfeatures)
#clf.fit(Xtrain,Ytrain)

#print clf.score(Xtest,Ytest)
#print "Random Forests accuracy"

#clf = ensemble.GradientBoostingClassifier()
#clf.fit(Xtrain,Ytrain)

#print "Gradient Boosting accuracy"
#print clf.score(Xtest,Ytest)

#clf = linear_model.LogisticRegression()
#clf.fit(Xtrain,Ytrain)

#print "Logistic Regression"
#print clf.score(Xtest,Ytest)

############################
# LOOCV UK and JP data
############################
looUK = cross_validation.LeaveOneOut(pd.X_UK.shape[0])
looJP = cross_validation.LeaveOneOut(pd.X_JP.shape[0])
looAll = cross_validation.LeaveOneOut(pd.X_JP.shape[0]+pd.X_UK.shape[0])

#X_UK = preprocessing.scale(pd.X_UK)
#X_JP = preprocessing.scale(pd.X_JP)
X_UK = pd.X_UK
X_JP = pd.X_JP
XAll = np.vstack((X_UK,X_JP))
YAll = np.hstack((pd.Y_UK,pd.Y_JP))

#XAll = preprocessing.scale(XAll, axis=1)

nsite1 = pd.X_UK.shape[0]
nsite2 = pd.X_JP.shape[0]
nAll = nsite1 + nsite2
nfeatures = pd.X_UK.shape[1]

Ysite = np.zeros(nsite1 + nsite2)
Ysite[pd.X_UK.shape[0]::]=1

#clf = ensemble.RandomForestClassifier(n_estimators=500, max_features = nfeatures)
#clf = ensemble.RandomForestClassifier(n_estimators=200)
#clf = svm.SVC(kernel='linear')
#
#sc = []
#for train, test in looUK:
#    X_train, X_test = pd.X_UK[train,], pd.X_UK[test,]
#    scaler = preprocessing.StandardScaler().fit(X_train)
#    X_train = scaler.transform(X_train)
#    X_test = scaler.transform(X_test)
#    Y_train, Y_test = pd.Y_UK[train,], pd.Y_UK[test,]
#    clf.fit(X_train,Y_train)
#    sc.append(clf.score(X_test,Y_test))
#    print test
#
#print "LOO-CV: UK"
#print np.asarray(sc).mean()
#
#sc = []
#for train, test in looUK:
#    X_train, X_test = pd.X_JP[train,], pd.X_JP[test,]
#    scaler = preprocessing.StandardScaler().fit(X_train)
#    X_train = scaler.transform(X_train)
#    X_test = scaler.transform(X_test)
#    Y_train, Y_test = pd.Y_JP[train,], pd.Y_JP[test,]       
#    clf.fit(X_train,Y_train)
#    sc.append(clf.score(X_test,Y_test))
#    print test
#
#print "LOO-CV: JP"
#print np.asarray(sc).mean()
#
#sc = []
#for train, test in looAll:
#    X_train, X_test = XAll[train,], XAll[test,]
#    scaler = preprocessing.StandardScaler().fit(X_train)
#    X_train = scaler.transform(X_train)
#    X_test = scaler.transform(X_test)
#    Y_train, Y_test = YAll[train,], YAll[test,]       
#    clf.fit(X_train,Y_train)
#    sc.append(clf.score(X_test,Y_test))
#    print test
#
#print "LOO-CV: All"
#print np.asarray(sc).mean()

############################
# Classify site (JP vs UK)
############################

#sc = []
#clf = ensemble.RandomForestClassifier(n_estimators=500)
#clf = ensemble.RandomForestClassifier(n_estimators=200)
#clf = svm.SVC(kernel='linear')
#for train, test in looAll:
#    X_train, X_test = XAll[train,], XAll[test,]
#    
##    if (test[0] < nsite1):
##        scaler1 = preprocessing.StandardScaler().fit(X_train[0:(nsite1-1),])
##        scaler2 = preprocessing.StandardScaler().fit(X_train[(nsite1-1)::,])
##        X_train[0:(nsite1-1),] = scaler1.transform(X_train[0:(nsite1-1),])
##        X_train[(nsite1-1)::,] = scaler2.transform(X_train[(nsite1-1)::,])
##        X_test = scaler1.transform(X_test)
##    else:
##        scaler1 = preprocessing.StandardScaler().fit(X_train[0:nsite1,])
##        scaler2 = preprocessing.StandardScaler().fit(X_train[nsite1::,])
##        X_train[0:nsite1,] = scaler1.transform(X_train[0:nsite1,])
##        X_train[nsite1::,] = scaler2.transform(X_train[nsite1::,])
##        X_test = scaler2.transform(X_test)
#        
#    scaler = preprocessing.StandardScaler().fit(X_train)
#    X_train = scaler.transform(X_train)
#    X_test = scaler.transform(X_test)
#    Y_train, Y_test = Ysite[train,], Ysite[test,]       
#    clf.fit(X_train,Y_train)
#    sc.append(clf.score(X_test,Y_test))
#    print test
#
#print "LOO-CV: All (sites)"
#print np.asarray(sc).mean()
    
#########################################
# T-test feature selection - Random Forests
#########################################

sc = []
se = []
sp = []
clf = ensemble.RandomForestClassifier(n_estimators=500, class_weight="auto" )
for train, test in looAll:
    
    X_train, X_test = XAll[train,], XAll[test,]
    Y_train, Y_test = YAll[train,], YAll[test,] 
    
    if (test[0] < nsite1):
        scaler1 = preprocessing.StandardScaler().fit(X_train[0:(nsite1-1),])
        scaler2 = preprocessing.StandardScaler().fit(X_train[(nsite1-1)::,])
        X_train[0:(nsite1-1),] = scaler1.transform(X_train[0:(nsite1-1),])
        X_train[(nsite1-1)::,] = scaler2.transform(X_train[(nsite1-1)::,])
        X_test = scaler1.transform(X_test)
    else:
        scaler1 = preprocessing.StandardScaler().fit(X_train[0:nsite1,])
        scaler2 = preprocessing.StandardScaler().fit(X_train[nsite1::,])
        X_train[0:nsite1,] = scaler1.transform(X_train[0:nsite1,])
        X_train[nsite1::,] = scaler2.transform(X_train[nsite1::,])
        X_test = scaler2.transform(X_test)
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #T-test 
    tt_ind = stats.ttest_ind(XAll[Y_train==1,], XAll[Y_train==0,])[1]<0.05   
      
    nfeat = sum(tt_ind)
    clf = ensemble.RandomForestClassifier(n_estimators=500, class_weight="auto" )
      
    #Fit model  
    clf.fit(X_train[::,tt_ind],Y_train)
    # Append accuracy
    sc_tmp = clf.score(X_test[::,tt_ind],Y_test)
    sc.append(sc_tmp)
    
    if Y_test == 1:
        se.append(sc_tmp)
    else:
        sp.append(sc_tmp)
    
    print test

print "LOO-CV: All (feature selection)"
print np.asarray(sc).mean()
print np.asarray(se).mean()
print np.asarray(sp).mean()

#########################################
# T-test feature selection - Grid Search SVM
#########################################

#sc = []
#clf = svm.SVC(kernel='linear')
#parameters = [{'kernel':['linear'],'C':[0.0001, 0.001, 0.01]}]
#for train, test in looAll:
#    
#    X_train, X_test = XAll[train,], XAll[test,]
#    Y_train, Y_test = YAll[train,], YAll[test,] 
#    
#    scaler = preprocessing.StandardScaler().fit(X_train)
#    X_train = scaler.transform(X_train)
#    X_test = scaler.transform(X_test)
#    
#    #T-test 
#    stats.ttest_ind(XAll[Y_train==1,], XAll[Y_train==0,])
#    tt_ind = stats.ttest_ind(XAll[Y_train==1,], XAll[Y_train==0,])[1]<0.05 
#    
#    clf_grid = GridSearchCV(clf, parameters, cv=10)
#    clf_grid.fit(X_train[::,tt_ind],Y_train)
#    
#    clf = svm.SVC(kernel='linear',C=clf_grid.best_params_['C'])
#    print clf_grid.best_params_['C']
#      
#    clf.fit(X_train[::,tt_ind],Y_train)
#    sc.append(clf.score(X_test[::,tt_ind],Y_test))
#    print test
#
#print "LOO-CV: All (feature selection)"
#print np.asarray(sc).mean()

#########################################
# T-test feature selection - Grid Search PCA
#########################################

#sc = []
#clf = ensemble.RandomForestClassifier(n_estimators=500)
#svr = svm.SVC()
#pca = decomposition.PCA()
#n_components=[10,20,30,40]
#pipe = Pipeline(steps=[('pca',pca),('classifier',svr)])
#
#for train, test in looAll:
#    
#    X_train, X_test = XAll[train,], XAll[test,]
#    Y_train, Y_test = YAll[train,], YAll[test,] 
#    
#    scaler = preprocessing.StandardScaler().fit(X_train)
#    X_train = scaler.transform(X_train)
#    X_test = scaler.transform(X_test)
#    
#    #PCA
#    clf_grid = GridSearchCV(pipe, dict(pca__n_components=n_components))
#    clf_grid.fit(X_train,Y_train)
#      
#    feat_select = decomposition.PCA(n_components=clf_grid.best_params_['pca__n_components']) 
#    X_train = feat_select.fit_transform(X_train)
#    X_test = feat_select.transform(X_test)
#    
#    print X_train.shape[1]
#    
#    clf.fit(X_train,Y_train)
#    sc.append(clf.score(X_test,Y_test))
#    print test
#
#print "LOO-CV: All (feature selection)"
#print np.asarray(sc).mean()