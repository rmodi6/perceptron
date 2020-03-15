import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MAX_ITERATIONS = 100_000


def train(X, y):
    w = np.zeros((X.shape[1],))
    i = 0
    while i < MAX_ITERATIONS:
        dot_product = y * np.dot(X, w)
        if np.any(dot_product <= 0):
            idx = np.where(dot_product <= 0)[0][0]
            w = w + y[idx] * X[idx]
            i += 1
        else:
            break
    return w


def compute_error(X, y, w):
    return len(np.where(y * np.dot(X, w) <= 0)[0]) / X.shape[0]


if __name__ == '__main__':
    # Argument parser to parse command line arguments
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
        s = int(m / k)
        batches = []

        for i in range(k):
            start_index, end_index = s * i, s * (i + 1)
            if i < (k - 1):
                batches.append((X[start_index:end_index], y[start_index:end_index]))
            else:
                batches.append((X[start_index:], y[start_index:]))

        weights, errors = [], []
        for i in range(k):
            train_X, train_y, test_X, test_y = None, None, None, None
            for j, (X, y) in enumerate(batches):
                if j == i:
                    test_X, test_y = X, y
                else:
                    if train_X is None:
                        train_X, train_y = X, y
                    else:
                        train_X, train_y = np.append(train_X, X, axis=0), np.append(train_y, y)
            weights.append(train(train_X, train_y))
            errors.append(compute_error(test_X, test_y, weights[-1]))
        print('Weights: %s \nErrors: %s \nMean Error: %s' % (weights, errors, np.mean(errors)))
    else:
        print('Incorrect mode of operation. Use "erm" or "cv".')
