import numpy as np


def least_square(feature, label):
    return (feature.T * feature).I * feature.T * label


def first_derivative(feature, label, w):
    m, n =np.shape(feature)
    g= np.mat(np.zeros((n,1)))
    for i in range(m):
        err = label[i,0] - feature[i,] * w
        for j in range(n):
            g[j,] -= err* feature[i,j]
    return g


def second_derivative(feature):
    m, n = np.shape(feature)
    G = np.mat(np.zeros((n, n)))
    for i in range(m):
        x_left = feature[i,].T
        x_right = feature[i,]
        G += x_left*x_right
    return G


def get_error(feature, label, w):
    return (label - feature * w).T * (label - feature * w)/2


def get_min_m(feature, label, sigma, delta, d, w, g):
    m = 0
    while True:
        w_new = w + pow(sigma, m) * d
        left = get_error(feature, label, w_new)
        right = get_error(feature, label, w) + delta * pow(sigma, m) * g.T * d
        if left <= right:
            break
        else:
            m += 1
    return m


def newton(feature, label, maxIter, sigma, delta):
    n = np.shape(feature)[1]
    w = np.mat(np.zeros((n,1)))
    iter = 0
    while iter < maxIter:
        print("\t---Iter: ", iter)
        g = first_derivative(feature, label, w)
        G = second_derivative(feature)
        d = -G.I * g
        m = get_min_m(feature, label, sigma, delta, d, w, g)
        w = w + pow(sigma, m) * d
        if iter % 10 == 0:
            print("\t---Iter: %d, error: %.3f" %(iter, get_error(feature, label, w)[0,0]))
        iter += 1
    return w


