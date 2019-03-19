import numpy as np


def distance(vecA, vecB):
    dist = (vecA - vecB) * (vecA - vecB).T
    return dist[0, 0]


def randCent(data, k):
    n = np.shape(data)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(data[:, j])
        rangeJ = np.max(data[:, j]) - minJ
        centroids[:, j] = minJ * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * rangeJ
    return centroids


def kmeans(data, k, centroids):
    m, n = np.shape(data)
    subCenter = np.mat(np.zeros((m, 2)))
    change = True
    while change == True:
        change = False
        for i in range(m):
            minDist = np.inf
            minIndex = 0
            for j in range(k):
                dist = distance(data[i,], centroids[j,])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            if subCenter[i, 0] != minIndex:
                change = True
                subCenter[i,] = np.mat([minIndex, minDist])
        for j in range(k):
            sum_all = np.mat(np.zeros((1, n)))
            r = 0
            for i in range(m):
                if subCenter[i, 0] == j:
                    sum_all += data[i,]
                    r +=1
            for z in range(n):
                try:
                    centroids[j, z] = sum_all[0, z] / r
                except:
                    print('r equals 0')
    return centroids, subCenter


def load_data(file_path):
    f = open(file_path, 'r')
    data = []
    for line in f.readlines():
        row = []
        lines = line.strip().split('\t')
        for x in lines:
            row.append(float(x))
        data.append(row)
    f.close()
    return np.mat(data)


def save_result(file_name, source):
    m, n = np.shape(source)
    f = open(file_name, 'w')
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(str(source[i, j]))
        f.write('\t'.join(tmp) + '\n')
    f.close()


def main():
    k = 4
    file_path = '../../../train_data/KMeans.train'
    data = load_data(file_path)
    centroids = randCent(data, k)
    centroids, subCenter = kmeans(data, k, centroids)
    save_result('../../../model/model.km.sub', subCenter)
    save_result('../../../model/model.km.center', centroids)


if __name__ == '__main__':
    main()
