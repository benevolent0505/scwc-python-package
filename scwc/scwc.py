import os
import subprocess

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from datetime import datetime

from sklearn.datasets import load_svmlight_file
from .utils import save_as_libsvm


class SCWC(BaseEstimator, TransformerMixin):

    def __init__(self, sort='mi', verbose=0):
        self._check_sort(sort)

        self.sort = sort
        self.verbose = verbose

        self._scwc = '{}/scwc_base/bin/scwc_base.jar'.format(os.path.dirname(os.path.realpath(__file__)))

        timestamp = datetime.now().strftime('%s')
        self._input_filename = 'input_{}.libsvm'.format(timestamp)
        self._output_filename = 'selected_features_{}.libsvm'.format(timestamp)

    def _check_sort(self, sort_measure):
        if sort_measure not in ['mi', 'su', 'icr', 'mcc']:
            os.exit(1)  # TODO: Exception

    def fit(self, X, y, header=None):
        self._file_dir = os.getcwd()
        save_as_libsvm(X, y, self._input_filename)

        return self

    def transform(self, X=None):
        options = '-o'
        if self.verbose > 0:
            options = '{}v'.format(options)

        subprocess.run(
            args=['java', '-jar', '-Xms2048m', self._scwc,
                  '-s', self.sort,
                  options,
                  '{}/{}'.format(self._file_dir, self._input_filename),
                  '{}/{}'.format(self._file_dir, self._output_filename)])
        subprocess.run(args=['rm', '{}/{}'.format(self._file_dir, self._input_filename)])

        X, _ = load_svmlight_file('{}/{}'.format(self._file_dir, self._output_filename))

        return X.tolil()

    def fit_transform(self, X, y, header=None):
        return self.fit(X, y, header).transform()
