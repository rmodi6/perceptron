import argparse
import random

import numpy as np
import pandas as pd

MAX_ITERATIONS = 100_000


def train(temp_X, temp_y):
    w = np.zeros((temp_X.shape[1],))
    i = 0
    while i < MAX_ITERATIONS:
        dot_product = temp_y * np.dot(temp_X, w)
        if np.any(dot_product <= 0):
            idx = np.where(dot_product <= 0)[0][0]
            w = w + temp_y[idx] * temp_X[idx]
            i += 1
        else:
            break
    return w


def compute_error(temp_X, temp_y, w):
    return len(np.where(temp_y * np.dot(temp_X, w) <= 0)[0]) / temp_X.shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset_path', action='store', type=str, help='path to dataset')
    parser.add_argument('--mode', dest='mode', action='store', type=str, help='mode of algorithm', default='erm')

    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)

    y = df.iloc[:, -1].values
    y = np.where(y == 0, -1, 1)

    X = df.iloc[:, :-1]
    X['bias'] = 1
    X = X.values

    if args.mode == 'erm':
        weight = train(X, y)
        error = compute_error(X, y, weight)
        print('Weights: %s \nError: %f' % (weight, error))
    elif args.mode == 'cv':
        m, d = X.shape
        k = 10
        s = int(m / k) + (1 if m % k != 0 else 0)
        batches = []

        indexes = list(range(X.shape[0]))
        random.shuffle(indexes)

        X = X[indexes]
        y = y[indexes]

        for i in range(k):
            start_index, end_index = s * i, s * (i + 1)
            batches.append((X[start_index:end_index], y[start_index:end_index]))

        weights, errors = [], []
        for i in range(k):
            print('Executing Fold #: %d' % (i + 1))
            train_X, train_y, test_X, test_y = None, None, None, None
            for j, (X, y) in enumerate(batches):
                if j == i:
                    test_X, test_y = X, y
                else:
                    if train_X is None:
                        train_X, train_y = X, y
                    else:
                        train_X, train_y = np.append(train_X, X, axis=0), np.append(train_y, y, axis=0)
            weights.append(train(train_X, train_y))
            errors.append(compute_error(test_X, test_y, weights[-1]))
            print('Weight: %s \nError: %s' % (weights[-1], errors[-1]))
        print('Errors: %s \nMean Error: %s' % (errors, np.mean(errors)))
    else:
        print('Incorrect mode of operation. Use "erm" or "cv".')
