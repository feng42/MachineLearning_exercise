# coding:UTF-8
import numpy as np
from ridge_regression import *

def load_data(file_path):
    f = open(file_path)
    feature = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split('\t')
        feature_tmp.append(1)
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        label.append(float(lines[-1]))
    f.close()
    return np.mat(feature), np.mat(label).T


def save_model(file_name, w):
    m, n =np.shape(w)
    with open(file_name, 'w') as fw:
        for i in range(m):
            w_tmp = []
            for j in range(n):
                w_tmp.append(str(w[i,j]))
            fw.write('\t'.join(w_tmp)+'\n')

def main():
    feature, label = load_data('../../../train_data/ridge_regression.train')
    method = 'bfgs'
    if method == 'bfgs':
        w0 = bfgs(feature, label, 0.5, 1000)
    elif method == 'lbfgs':
        w0 = lbfgs(feature, label, 0.5, 1000, m=10)
    else:
        print('Perhaps should use linear regression')
    try:
        save_model('../../../model/model.ridge', w0)
    except:
        print('Perhaps no weights available')
    else:
        print('Weights saved')

if __name__ == "__main__":
    main()