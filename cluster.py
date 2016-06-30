from sklearn import cluster
from processing import *


def cluster_data(X, Y, n_clusters=6):
	clusterized_X, clusterized_Y = [], []
	for i in range(n_clusters):
		clusterized_X.append([])
		clusterized_Y.append([])

	print 'Start clustering...'
	algorithm = cluster.MiniBatchKMeans(n_clusters=6)
	algorithm.fit(X)
	print 'Start predicting clusters...'
	data_clusters = [algorithm.predict(x)[0] for x in X]
	print len(data_clusters)
	print data_clusters[:10]
	print 'Start saving...'
	for cluster_num, data_point, y in zip(data_clusters, X, Y):
		clusterized_X[cluster_num].append(data_point)
		clusterized_Y[cluster_num].append(y)	
	return algorithm, clusterized_X, clusterized_Y
	 