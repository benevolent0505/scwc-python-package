import numpy as np
from scipy.sparse import issparse


def save_as_libsvm(X, y, filename):
    with open(filename, 'w') as file:
        if issparse(X):
            for i, label in enumerate(y):
                file.write('{} {}\n'.format(label, _sparse_arr2libsvm(X[i])))
        else:
            for i, label in enumerate(y):
                file.write('{} {}\n'.format(label, _arr2libsvm(X[i])))


def _arr2libsvm(arr):
    non_zero_index = list(np.nonzero(arr)[0])
    non_zero_value = arr[np.nonzero(arr)]

    return get_libsvm_str(non_zero_index, non_zero_value)


def _sparse_arr2libsvm(arr):
    _, nonzero_indexes = arr.nonzero()
    nonzero_indexes.sort()
    nonzero_values = [arr[0, i] for i in nonzero_indexes]

    return get_libsvm_str(nonzero_indexes, nonzero_values)


def get_libsvm_str(nonzero_indexes, nonzero_values):
    libsvm_str = ['{}:{}'.format(i + 1, v) for (i, v) in zip(nonzero_indexes, nonzero_values)]
    return ' '.join(libsvm_str)
