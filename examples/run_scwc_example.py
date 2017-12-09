#!/usr/bin/env python

from scwc import SCWC

from sklearn.datasets import load_wine


wine = load_wine()
X = wine.data
y = wine.target
header = wine.feature_names

selector = SCWC(verbose=10)
selector.fit(X, y, header=header)

selected_feature = selector.transform()
