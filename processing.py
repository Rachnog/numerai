# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:37:40 2016

@author: Alex
"""

import numpy as np
import glob, itertools
from sklearn import preprocessing
import csv


def load_probs(filename):
    f = open(filename, 'rb').readlines()[1:]
    features, targets = [], []
    for line in f:
        raw = [float(s) for s in line.split(',')]
        features.append(raw[0])
        targets.append(raw[1])
    return features, targets


def save_probs(filename, ids, probs):
    data = []
    for i, p in zip(ids, probs):
        data.append([int(i), p])

    writer = csv.writer(open(filename, 'wb'))
    writer.writerow(['t_id','probability'])
    writer.writerows(data)
    return data    


def load_as_ts(n_features=21):
    f = open('numerai_training_data.csv', 'rb').readlines()[1:]
    features, targets = [], []
    for line in f:
        raw = [float(s) for s in line.split(',')]
        features.append(raw[:-1][:n_features-1])
        targets.append(raw[-2])
    return features, targets


def change_submission(filename, newfilename):
    f = open(filename, 'rb').readlines()[1:]
    data = []
    for line in f:
        index, prob = int(line.split(',')[0]), float(line.split(',')[1])
        if prob > 0.5:
            prob += 0.05
        else:
            prob -= 0.05
        data.append([index, prob])

    writer = csv.writer(open(newfilename, 'wb'))
    writer.writerow(['t_id','probability'])
    writer.writerows(data)
    return data


def load_old_data(n_features=21):
    f = open('numerai_training_data2.csv', 'rb').readlines()[1:]
    features, targets = [], []
    for line in f:
        raw = [float(s) for s in line.split(',')]
        features.append(raw[:-1][:n_features])
        targets.append(int(raw[-1]))
    return features, targets


def load_data(n_features=21):
    f = open('numerai_training_data.csv', 'rb').readlines()[1:]
    features, targets = [], []
    for line in f:
        raw = [float(s) for s in line.split(',')]
        features.append(raw[:-1][:n_features])
        targets.append(int(raw[-1]))
    return features, targets


def load_test_data():
    f = open('numerai_tournament_data.csv', 'rb').readlines()[1:]
    features, ids = [], []
    for line in f:
        raw = [float(s) for s in line.split(',')[1:]]
        features.append(raw)
        ids.append(int(line.split(',')[0]))
    return features, ids


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def create_Xt_Yt(X, y, percentage=0.8):
    X, y = shuffle_in_unison(X, y)

    X_train = X[0:len(X) * percentage]
    Y_train = y[0:len(y) * percentage]

    X_test = X[len(X) * percentage:]
    Y_test = y[len(X) * percentage:]

    return X_train, X_test, Y_train, Y_test
