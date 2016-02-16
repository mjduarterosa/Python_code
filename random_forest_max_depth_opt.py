import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import load_pain_data as pd

X = pd.X_JP
y = pd.Y_JP

max_estimators = 1000
oob = True
score_d1 = [0]*max_estimators
score_d2 = [0]*max_estimators
score_d3 = [0]*max_estimators
score_d5 = [0]*max_estimators
score_d10 = [0]*max_estimators
score_dN = [0]*max_estimators
q = 0
for i in range(0,max_estimators,5):
    clf = RandomForestClassifier(n_estimators = (i+1), max_depth = 1, oob_score = oob)
    clf.fit(X, y)
    score_d1[q] = (1 - clf.oob_score_)
    clf = RandomForestClassifier(n_estimators = (i+1), max_depth = 2, oob_score = oob)
    clf.fit(X, y)
    score_d2[q] = (1 - clf.oob_score_)
    clf = RandomForestClassifier(n_estimators = (i+1), max_depth = 3, oob_score = oob)
    clf.fit(X, y)
    score_d3[q] = (1 - clf.oob_score_)
    clf = RandomForestClassifier(n_estimators = (i+1), max_depth = 5, oob_score = oob)
    clf.fit(X, y)
    score_d5[q] = (1 - clf.oob_score_)
    clf = RandomForestClassifier(n_estimators = (i+1), max_depth = 10, oob_score = oob)
    clf.fit(X, y)
    score_d10[q] = (1 - clf.oob_score_)
    clf = RandomForestClassifier(n_estimators = (i+1), max_depth = None, oob_score = oob)
    clf.fit(X, y)
    score_dN[q] = (1 - clf.oob_score_)
    q = q+1
    print q

with plt.style.context('fivethirtyeight'):
    plt.plot(range(1,max_estimators,5),score_d1[0:200],linewidth=1)
    plt.plot(range(1,max_estimators,5),score_d2[0:200],linewidth=1)
    plt.plot(range(1,max_estimators,5),score_d3[0:200],linewidth=1)
    plt.plot(range(1,max_estimators,5),score_d5[0:200],linewidth=1)
    plt.plot(range(1,max_estimators,5),score_d10[0:200],linewidth=1)
    plt.plot(range(1,max_estimators,5),score_dN[0:200],linewidth=1)

plt.show()