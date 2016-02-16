# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:32:37 2016

@author: mariarosa
"""

# PCA code
randomforests = ensemble.RandomForestClassifier(n_estimators=200)
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca),('randomforests', randomforests)])

n_components = [10, 20, 30, 40, 50]
estimator = GridSearchCV(pipe, dict(pca__n_components=n_components))
estimator.fit(XAll, YAll)