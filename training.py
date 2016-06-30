# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:47:27 2016

@author: Alex
"""

from processing import *
from cluster import *

import numpy as np
import matplotlib.pyplot as plt
import csv
import cPickle
import glob

'''
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D

from keras.layers.recurrent import LSTM
'''

def make_subm(model, X_sub, ids, sub_name):
	data = []
	preds = model.predict(np.array(X_sub))
	for x, i in zip(preds, ids):
		line = [i, x[0]]
		data.append(line)

	writer = csv.writer(open(sub_name, 'wb'))
	writer.writerow(['t_id','probability'])
	writer.writerows(data)
	return data


def make_subm_cluster(models, c_algorithm, X_sub, ids, sub_name):
	data = []
	X_sub = np.array(X_sub)
	X_sub2cnn = np.reshape(X_sub, (X_sub.shape[0], X_sub.shape[1], 1))
	for x, xc, i in zip(X_sub, X_sub2cnn, ids):
		cluster_num = c_algorithm.predict(x)[0]
		prediction = models[cluster_num].predict(np.array([xc]))
		line = [i, prediction[0][0]]
		print cluster_num, line
		data.append(line)

	writer = csv.writer(open(sub_name, 'wb'))
	writer.writerow(['t_id','probability'])
	writer.writerows(data)
	return data


def make_reverse_subm(model, X_sub, ids, sub_name):
	data = []
	preds = model.predict(np.array(X_sub))
	for x, i in zip(preds, ids):
		line = [i, 1 - x[0]]
		data.append(line)

	writer = csv.writer(open(sub_name, 'wb'))
	writer.writerow(['t_id','probability'])
	writer.writerows(data)
	return data


def trainRNN(X_train, X_test, Y_train, Y_test):
	print 'Building model...'
	model = Sequential()
	model.add(LSTM(input_shape = (1,), input_dim=1, output_dim=50, return_sequences=True))
	model.add(LSTM(input_shape = (1,), input_dim=1, output_dim=50, return_sequences=False))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(optimizer='adam', 
				  loss='binary_crossentropy', 
				  metrics=['accuracy'])

	print 'Training model...'
	model.fit(X_train, 
		      Y_train, 
			  nb_epoch=5, 
			  batch_size = 128, 
			  verbose=1, 
			  validation_split=0.1)
	score = model.evaluate(X_test, Y_test, batch_size=128)
	print score
	return model	


def trainMLPregression(X_train, X_test, Y_train, Y_test):
	print 'Building model...'
	model = Sequential()
	model.add(Dense(500, input_shape = (len(X_train[0]), )))
	model.add(Activation('relu'))
	model.add(Dense(250))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('linear'))

	model.compile(optimizer='adam', 
				  loss='mse')

	print 'Training model...'
	model.fit(X_train, 
		      Y_train, 
			  nb_epoch=20, 
			  batch_size = 128, 
			  verbose=1, 
			  validation_split=0.1)
	score = model.evaluate(X_test, Y_test, batch_size=128)
	print score
	return model

def trainMLP(X_train, X_test, Y_train, Y_test):
	print 'Building model...'
	model = Sequential()
	model.add(Dense(500, input_shape = (len(X_train[0]), )))
	model.add(Activation('relu'))
	model.add(Dense(250))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(optimizer='adam', 
				  loss='binary_crossentropy', 
				  metrics=['accuracy'])

	print 'Training model...'
	model.fit(X_train, 
		      Y_train, 
			  nb_epoch=7, 
			  batch_size = 128, 
			  verbose=1, 
			  validation_split=0.1)
	score = model.evaluate(X_test, Y_test, batch_size=128)
	print score
	return model


def trainCNN(X_train, X_test, Y_train, Y_test, weights_path = None):
	print 'Building model...'
	model = Sequential()
	model.add(Convolution1D(input_shape = (len(X_train[0]), 1), 
	                        nb_filter=128,
	                        filter_length=2,
	                        border_mode='valid',
	                        activation='relu',
	                        subsample_length=1))
	model.add(MaxPooling1D(pool_length=2))

	model.add(Flatten())

	model.add(Dense(250))
	model.add(Activation('relu'))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(optimizer='adam', 
				  loss='binary_crossentropy', 
				  metrics=['accuracy'])

	print 'Training model...'
	model.fit(X_train, 
		      Y_train, 
			  nb_epoch=7, 
			  batch_size = 64, 
			  verbose=0, 
			  validation_split=0.1)
	score = model.evaluate(X_test, Y_test, batch_size=128)
	print score

	if weights_path != None:
		model.save_weights(weights_path)

	return model


def getCNN(weights_path):
	model = Sequential()
	model.add(Convolution1D(input_shape = (21, 1), 
	                        nb_filter=64,
	                        filter_length=2,
	                        border_mode='valid',
	                        activation='relu',
	                        subsample_length=1))
	model.add(MaxPooling1D(pool_length=2))

	model.add(Flatten())

	model.add(Dense(250))
	model.add(Activation('relu'))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(optimizer='adam', 
				  loss='binary_crossentropy', 
				  metrics=['accuracy'])

	model.load_weights(weights_path)	
	return model

'''
print 'Data processing...'  
X, Y = load_data()
X, Y = np.array(X), np.array(Y)
X_sub, ids = load_test_data()
'''

allprobs = []
for filename in glob.glob('./subm/*.csv'):
	print filename
	_, cnn_probs = load_probs(filename)
	allprobs.append(np.array(cnn_probs))

for a in allprobs:
	print a
	print '-' * 10
print '*' * 20

new_probs = np.zeros(len(allprobs[0]))
for a in allprobs:
	print new_probs.shape
	print a.shape
	new_probs += a

new_probs /= len(allprobs)
d = save_probs('all_together_average.csv', _, new_probs)



# HAND-CRAFTED ENSEMBLE
'''
_, cnn_probs = load_probs('./best/raw_features_cnn_best2.csv')
_, mlp_probs = load_probs('./best/raw_features_mlp_0.csv')
_, mlpt_probs = load_probs('./best/test_tree_features_mlp.csv')

cnn_probs = np.array(cnn_probs)
mlp_probs = np.array(mlp_probs)
mlpt_probs = np.array(mlpt_probs)

for i, (a, b, c) in enumerate(zip(cnn_probs, mlp_probs, mlpt_probs)):
	print a, b, c
	if i > 10:
		break


res_probs = 0.1 * cnn_probs + 0.1 * mlp_probs + 0.8 * mlpt_probs

d = save_probs('average2.csv', _, res_probs)
'''


# EXPERIMENTS WITH REGRESSION
'''
X, Y = load_as_ts()
X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)
model = trainMLPregression(X_train, X_test, Y_train, Y_test)


predicted = model.predict(X_test)
try:
    fig = plt.figure(figsize=(width, height))
    plt.plot(Y_test[:150], color='black')
    plt.plot(predicted[:150], color='blue')
    plt.show()
except Exception as e:
    print str(e)

#change_submission('raw_features_cnn_best2.csv', 'raw_features_cnn_best2_c2.csv')
'''


# SAVING CLUSTERIZER
'''
c_algorithm, CX, CY = cluster_data(X, Y)
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(c_algorithm, fid)    
'''


# EXPERIMENTS WITH CLUSTERING
'''
with open('my_dumped_classifier.pkl', 'rb') as fid:
    c_algorithm = cPickle.load(fid)

models = [getCNN('cnn_' + str(i) + '.hdf5') for i in range(6)]
make_subm_cluster(models, c_algorithm, X_sub, ids, 'clustered_cnns.csv')
'''

'''
c_algorithm, CX, CY = cluster_data(X, Y)
c_models = []
for i, (cluster_X, cluster_Y) in enumerate(zip(CX, CY)):
	cluster_X, cluster_Y = np.array(cluster_X), np.array(cluster_Y)
	X_train, X_test, Y_train, Y_test = create_Xt_Yt(cluster_X, cluster_Y)
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	model_cnn = trainCNN(X_train, X_test, Y_train, Y_test, 'cnn_' + str(i) + '.hdf5')
	print i, 'CNN raw trained'
	c_models.append(model_cnn)
'''


# RAW MLP
'''
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)
model_raw = trainMLP(X_train, X_test, Y_train, Y_test)
a = make_subm(model_raw, X_sub, ids, 'raw_features_mlp_' + str(1) + '.csv')
print 'MLP raw trained'
'''

# TREE FEATURE MLP
'''
clf = ExtraTreesClassifier()
clf = clf.fit(X, Y)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_sub = model.transform(X_sub)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X_new, Y)
model_tree = trainMLP(X_train, X_test, Y_train, Y_test)
a = make_subm(model_tree, X_sub, ids, 'test_tree_features_mlp.csv')
print 'MLP tree trained'
'''

# PCA MLP
'''
pca = PCA(n_components=16)
X_pca = pca.fit_transform(X)
X_sub = pca.fit_transform(X_sub)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X_pca, Y)
model_pca = trainMLP(X_train, X_test, Y_train, Y_test)
a = make_subm(model_pca, X_sub, ids, 'test_pca_features_mlp.csv')
print 'MLP pca trained'
'''

#RAW RNN
'''
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_sub = np.array(X_sub)
X_sub = np.reshape(X_sub, (X_sub.shape[0], X_sub.shape[1], 1))
model_rnn = trainRNN(X_train, X_test, Y_train, Y_test)
a = make_subm(model_rnn, X_sub, ids, 'raw_features_rnn.csv')
print 'RNN raw trained'
'''

# LOTS OF CNNS
'''
for i in range(10):
	X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	X_sub = np.array(X_sub)
	X_sub = np.reshape(X_sub, (X_sub.shape[0], X_sub.shape[1], 1))
	model_cnn = trainCNN(X_train, X_test, Y_train, Y_test)
	a = make_subm(model_cnn, X_sub, ids, 'raw_features_cnn128_' + str(i) + '.csv')
	print 'CNN raw trained'
'''