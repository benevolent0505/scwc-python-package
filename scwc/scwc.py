import os
import subprocess

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from datetime import datetime


class SCWC(BaseEstimator, TransformerMixin):

    def __init__(self, sort='su', verbose=0):
        self._check_sort(sort)

        self.sort = sort
        self.verbose = verbose
        self.selected_feature_names = []

        self._scwc = '{}/scwc_base/bin/scwc_base.jar'.format(os.path.dirname(os.path.realpath(__file__)))

        timestamp = datetime.now().strftime('%s')
        self._input_filename = 'input_{}.csv'.format(timestamp)
        self._output_filename = 'selected_features_{}.csv'.format(timestamp)

    def _check_sort(self, sort_measure):
        if sort_measure not in ['mi', 'su', 'icr', 'mcc']:
            os.exit(1)  # TODO: Exception

    def fit(self, X, y, header=None):
        if not header:
            header = list(map(lambda x: 'L{}'.format(x + 1), range(X.shape[1])))

        self.data_dtype = X.dtype
        data = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
        df = pd.DataFrame(data, columns=header + ['class'])

        self._file_dir = os.getcwd()
        df.to_csv(self._input_filename, index=False)

        return self

    def transform(self, X=None):
        options = '-o'
        if self.verbose > 0:
            options = '{}v'.format(options)

        subprocess.run(
            args=['java', '-jar', self._scwc,
                  '-s', self.sort,
                  options,
                  '{}/{}'.format(self._file_dir, self._input_filename),
                  '{}/{}'.format(self._file_dir, self._output_filename)])
        subprocess.run(args=['rm', '{}/{}'.format(self._file_dir, self._input_filename)])

        X = pd.read_csv('{}/{}'.format(self._file_dir, self._output_filename))
        self.selected_feature_names = list(X.columns)[:-1]

        return np.array(X.values[:, :-1], dtype=self.data_dtype)

    def fit_transform(self, X, y, header=None):
        return self.fit(X, y, header).transform()
