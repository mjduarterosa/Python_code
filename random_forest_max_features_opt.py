import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import load_pain_data as pd

X = pd.X_JP
y = pd.Y_JP

# Parameter behaviour
# ---------------------------------------------------------------
# ---------------------------------------------------------------
max_estimators = 1000
oob = True
score_1 = [0]*max_estimators
score_n = [0]*max_estimators
score_a = [0]*max_estimators
score_l = [0]*max_estimators
score_h = [0]*max_estimators
q = 0
nfeatures = X.shape[1]
for i in range(0,max_estimators,5):
    clf = RandomForestClassifier(n_estimators = (i+1), max_features = 1, oob_score = oob)
    clf.fit(X, y)
    score_1[q] = (1 - clf.oob_score_)
    clf = RandomForestClassifier(n_estimators = (i+1), max_features = nfeatures, oob_score = oob)
    clf.fit(X, y)
    score_n[q] = (1 - clf.oob_score_)
    clf = RandomForestClassifier(n_estimators = (i+1), max_features = "auto", oob_score = oob)
    clf.fit(X, y)
    score_a[q] = (1 - clf.oob_score_)
    clf = RandomForestClassifier(n_estimators = (i+1), max_features = "log2", oob_score = oob)
    clf.fit(X, y)
    score_l[q] = (1 - clf.oob_score_)
    clf = RandomForestClassifier(n_estimators = (i+1), max_features = 4, oob_score = oob)
    clf.fit(X, y)
    score_h[q] = (1 - clf.oob_score_)
    q = q+1
    print q

with plt.style.context('fivethirtyeight'):
    plt.plot(range(1,max_estimators,5),score_1[0:200],linewidth=1)
    plt.plot(range(1,max_estimators,5),score_n[0:200],linewidth=1)
    plt.plot(range(1,max_estimators,5),score_a[0:200],linewidth=1)
    plt.plot(range(1,max_estimators,5),score_l[0:200],linewidth=1)
    plt.plot(range(1,max_estimators,5),score_h[0:200],linewidth=1)

plt.show()