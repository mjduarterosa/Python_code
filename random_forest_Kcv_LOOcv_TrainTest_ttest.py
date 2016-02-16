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
import scipy.stats as stats

#from sklearn.datasets import load_iris
#data = load_iris()
#X = data.data
#y = data.target

f = open('/Users/mariarosa/Downloads/Hiro_data.csv')
ncontrols = 14
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

Xc = X[0:ncontrols,::]
Xp = X[ncontrols:nsamples,::]

t, prob = stats.ttest_ind(Xc, Xp)

# k-fold CV Pipeline
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Random forest
clf = RandomForestClassifier(n_estimators = 1000, max_features = 1)

# LOO CV Pipeline
# ---------------------------------------------------------------
# ---------------------------------------------------------------
loo = cross_validation.LeaveOneOut(nsamples)
sc = 0
i = 0
for train, test in loo:
    print 'Fold: ' + str(test[0])
    
    # Feature selection (two-sample ttest)
    Xc = X[0:ncontrols,::]
    Xp = X[ncontrols:nsamples,::]
    
    if test[0] <= ncontrols-1:
        Xc = np.delete(Xc, test[0], 0)
    else:
        Xp = np.delete(Xp, test[0]-ncontrols, 0)
    
    t, prob = stats.ttest_ind(Xc, Xp)
    
    Xt = X[train,]
    yt = y[train,]

    Xte = X[test,]
    yte = y[test,]
    
    clf.fit(Xt[::,prob<0.05], yt)
    sc = sc + clf.score(Xte[::,prob<0.05], yte)

print 'LOO Accuracy:'
print sc/float(nsamples)


# Train/Test Pipeline
# ---------------------------------------------------------------
# ---------------------------------------------------------------
#clf.fit(X2, y2)
#print 'Train/Test Accuracy ADNI1:'
#print clf.score(X1, y1)




