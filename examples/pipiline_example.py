#!/usr/bin/env python

from scwc import SCWC
import numpy as np

from sklearn.datasets import load_wine

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score


wine = load_wine()

X = wine.data
y = wine.target
names = wine.feature_names

selector = SCWC(sort='su', verbose=1)
X_selected = selector.fit_transform(X, y, header=names)

pipe = Pipeline([('scaler', StandardScaler()),
                 ('esitomator', LogisticRegression(random_state=0))])

scores = cross_val_score(pipe, X_selected, y, scoring='accuracy', cv=10, verbose=10)
print('{:0.3f} +/- {:0.3f}'.format(np.mean(scores), np.std(scores)))
