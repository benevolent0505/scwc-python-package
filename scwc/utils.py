import numpy as np


def save_as_libsvm(X, y, filename):
    with open(filename, 'w') as file:
        for i, label in enumerate(y):
            file.write('{} {}\n'.format(label, _arr2libsvm(X[i])))


def _arr2libsvm(arr):
    non_zero_index = np.nonzero(arr)
    non_zero_value = arr[non_zero_index]

    libsvm_str = ['{}:{}'.format(i + 1, v) for (i, v) in zip(list(non_zero_index[0]), non_zero_value)]
    return ' '.join(libsvm_str)
