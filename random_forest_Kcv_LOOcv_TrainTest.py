# !/usr/bin/python

# Import packages
# ---------------------------------------------------------------
import numpy as np
import csv
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn import preprocessing

f = open('/Users/mariarosa/Downloads/Hiro_data.csv')
npatients = 17
csv_f = csv.reader(f)
data = []
for row in csv_f:
    data.append(row)
X = np.asarray(data, dtype = 'float')
X = preprocessing.scale(X, axis=1)

nsamples = np.shape(X)[0]
nfeatures = np.shape(X)[1]
y = [0 for i in range(nsamples)]
y[14::] = [1 for i in range(npatients)]
y = np.asarray(y, dtype = 'int')

# k-fold CV Pipeline
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Random forest
clf = RandomForestClassifier(n_estimators = 1000, max_features = 1)

# 10 fold CV
#sc = 0
#n_folds = 10
#kf = cross_validation.KFold(n_samples1, n_folds=n_folds)
#for train, test in kf:
#    clf.fit(X1[train,], y1[train])
#    sc = sc + clf.score(X1[test,], y1[test])
#print 'K-fold Accuracy - ADNI1'
#print sc/float(n_folds)

# LOO CV Pipeline
# ---------------------------------------------------------------
# ---------------------------------------------------------------
loo = cross_validation.LeaveOneOut(nsamples)
sc = 0
i = 0
for train, test in loo:
    print 'Fold: ' + str(i)
    clf.fit(X[train,], y[train])
    sc = sc + clf.score(X[test,], y[test])
    i = i + 1
print 'LOO Accuracy:'
print sc/float(nsamples)


# Train/Test Pipeline
# ---------------------------------------------------------------
# ---------------------------------------------------------------
#clf.fit(X2, y2)
#print 'Train/Test Accuracy ADNI1:'
#print clf.score(X1, y1)




