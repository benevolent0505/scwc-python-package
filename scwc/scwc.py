import os
import subprocess

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from datetime import datetime

from sklearn.datasets import dump_svmlight_file, load_svmlight_file


class SCWC(BaseEstimator, TransformerMixin):

    def __init__(self, sort='mi', verbose=0):
        self._check_sort(sort)

        self.sort = sort
        self.verbose = verbose

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self._scwc = os.path.abspath('{}/scwc_base/bin/scwc_base.jar'.format(cur_dir))

        timestamp = datetime.now().strftime('%s')
        self._input_filename = 'input_{}.libsvm'.format(timestamp)
        self._output_filename = 'selected_features_{}.libsvm'.format(timestamp)
        self._log_filename = 'selected_features_{}.log'.format(timestamp)

    def _check_sort(self, sort_measure):
        """Raises a ValueError if sort_measure is not known measure"""
        if sort_measure not in ['mi', 'su', 'icr', 'mcc']:
            raise ValueError('Unsupported sorting measure')

    def fit(self, X, y, header=None):
        dump_svmlight_file(X, y, self._input_filename, zero_based=False)

        options = '-vlo' if self.verbose > 0 else '-lo'
        inputfile = '{}/{}'.format(os.getcwd(), self._input_filename)
        outputfile = '{}/{}'.format(os.getcwd(), self._output_filename)
        try:
            subprocess.run(
                args=['java', '-Xmx8g', '-jar', self._scwc,
                      '-s', self.sort, options, inputfile, outputfile])
            self._feature_size = X.shape[1]
            self._outputfile = os.path.abspath(outputfile)
        except subprocess.CalledProcessError:
            # output error log
            pass
        finally:
            os.remove(os.path.abspath(inputfile))

        logfile = os.path.abspath('{}/{}'.format(os.getcwd(), self._log_filename))
        with open(logfile, 'r') as log:
            self._selected_indices = self._get_selected_indices(log)

        os.remove(logfile)

        return self

    def transform(self, X=None):
        if X is not None:
            X_selected = X[:, self.get_support()]
        else:
            X_selected, _ = load_svmlight_file(self._outputfile)
            os.remove(self._outputfile)

        return X_selected

    def fit_transform(self, X, y, header=None):
        return self.fit(X, y, header).transform()

    def get_support(self, indices=False):
        support = np.full(self._feature_size, False)
        support[self._selected_indices] = True

        return support if indices else self._selected_indices

    def _get_selected_indices(self, logfile):
        selected_indices = []
        for line in logfile:
            if line.startswith('#'):
                continue

            tmp_selected_index = line.split()[-1]
            if tmp_selected_index in ('feature', 'patchFeature'):
                continue

            selected_index = tmp_selected_index.strip('att_')
            selected_indices.append(int(selected_index) - 1)  # to zero_based index

        return selected_indices
