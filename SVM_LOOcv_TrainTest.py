# Compare with SVM
# ---------------------------------------------------------------
# ---------------------------------------------------------------
X1 = preprocessing.scale(X1)
X2 = preprocessing.scale(X2)

score_cv = []
c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
for i in c_values:
    clf_svm = svm.SVC(kernel='linear', C = i)
    sc = 0
    loo = cross_validation.LeaveOneOut(np.shape(X)[0])
    for train, test in loo:
        clf_svm.fit(X1[train,], y1[train])
        sc = sc + clf_svm.score(X1[test, ], y1[test])
    score_cv = np.append(score_cv, float(sc)/float(np.shape(X1)[0]))
mx = [i for i,j in enumerate(score_cv) if j == max(score_cv)]
print c_values[mx[0]]

clf_svm = svm.SVC(kernel='linear', C = c_values[mx[0]])

clf_svm.fit(X1, y1)

print clf_svm.score(X2,y2)

score_cv = []
c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
for i in c_values:
    clf_svm = svm.SVC(kernel='linear', C = i)
    sc = 0
    loo = cross_validation.LeaveOneOut(np.shape(X)[0])
    for train, test in loo:
        clf_svm.fit(X2[train,], y2[train])
        sc = sc + clf_svm.score(X2[test, ], y2[test])
    score_cv = np.append(score_cv, float(sc)/float(np.shape(X2)[0]))
mx = [i for i,j in enumerate(score_cv) if j == max(score_cv)]
print c_values[mx[0]]

clf_svm = svm.SVC(kernel='linear', C = c_values[mx[0]])

clf_svm.fit(X2, y2)

print clf_svm.score(X1,y1)