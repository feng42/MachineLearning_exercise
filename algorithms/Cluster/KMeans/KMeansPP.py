import numpy as np
from random import random
from KMeans import load_data,kmeans,distance,save_result

FLOAT_MAX = 1e100


def nearest(point, cluster_centers):
    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]
    for i in range(m):
        d = distance(point, cluster_centers[i, ])
        if min_dist > d:
            min_dist = d
    return min_dist


def getCentroids(points, k):
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k, n)))
    index = np.random.randint(0, m)
    cluster_centers[0, ] = np.copy(points[index, ])
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            d[j] = nearest(points[j, ], cluster_centers[0:i, ])
            sum_all += d[j]
        sum_all *= random()
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j, ])
            break
    return cluster_centers


if __name__ == '__main__':
    k = 4
    file_path = '../../../train_data/KMeans.train'
    data = load_data(file_path)
    centroids = getCentroids(data, k)
    _, subCenter  = kmeans(data, k, centroids)
    save_result('../../../model/model.kmcpp.sub', subCenter)
    save_result('../../../model/model.kmcpp.center', centroids)
