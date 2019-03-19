# coding:UTF-8
import  numpy as np


def load_data(file_path):
    f = open(file_path)
    feature = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split('\t')
        feature_tmp.append(1)
        for i in range(len(lines)):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
    f.close()
    return np.mat(feature)


def load_model(model_file):
    w = []
    f = open(model_file)
    for line in f.readlines():
        w.append(float(line.strip()))
    f.close()
    return np.mat(w).T


def get_prediction(data, w):
    return data*w


if __name__ == "__main__":
    testData = load_data('../../test_data/linear_regression.test')
    w =load_model('../../model/model.linearR')
    predict = get_prediction(testData, w)
    print(predict)
