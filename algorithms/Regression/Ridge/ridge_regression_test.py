# coding:UTF-8

import numpy as np


def load_data(file_path):
    f = open(file_path)
    feature = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)  # x0
        for i in xrange(len(lines)):
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
    return data * w


def save_result(file_name, predict):
    m = np.shape(predict)[0]
    result = []
    for i in xrange(m):
        result.append(str(predict[i, 0]))
    f = open(file_name, "w")
    f.write("\n".join(result))
    f.close()


if __name__ == "__main__":
    # 1、导入测试数据
    print "----------1.load data ------------"
    testData = load_data("../../../test_data/ridge_regression.test")
    # 2、导入线性回归模型
    print "----------2.load model ------------"
    w = load_model("../../../model/model.ridge")
    # 3、得到预测结果
    print "----------3.get prediction ------------"
    predict = get_prediction(testData, w)
    # 4、保存最终的结果
    print "----------4.save prediction ------------"
    save_result("../../../result/res.ridge", predict)