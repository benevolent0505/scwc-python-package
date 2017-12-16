#!/usr/bin/env python

from scwc import SCWC

from sklearn.datasets import load_wine
from scipy.sparse import lil_matrix


wine = load_wine()
X = wine.data
y = wine.target

selector = SCWC(verbose=10)
selector.fit(X, y)

selected_feature = selector.transform()


# sparse
X = lil_matrix(X)

selector = SCWC(sort='icr', verbose=1)
selector.fit_transform(X, y)
