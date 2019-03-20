import numpy as np
import random_forest as rdf
import cPickle as pickle

def load_data(file_name):
    data_train = []
    f = open(file_name)
    for line in f.readlines():
        lines = line.strip().split("\t")
        data_tmp = []
        for x in lines:
            data_tmp.append(float(x))
        data_train.append(data_tmp)
    f.close()
    return data_train


def get_predict(trees_result, trees_feature, data_train):
    m_tree = len(trees_result)
    m = np.shape(data_train)[0]
    result = []
    for i in range(m_tree):
        clf = trees_result[i]
        feature = trees_feature[i]
        data = split_data(data_train, feature)
        result_i = []
        for i in range(m):
            result_i.append((rdf.predict(data[i][0:-1], clf).keys())[0])
        result.append(result_i)
    final_predict = np.sum(result, axis=0)
    return final_predict


def split_data(data_train, feature):
    m = np.shape(data_train)[0]
    data = []
    for i in range(m):
        data_x_tmp = []
        for x in feature:
            data_x_tmp.append(data_train[i][x])
        data_x_tmp.append(data_train[i][-1])
        data.append(data_x_tmp)
    return data


def calc_correct_rate(data_train, final_predict):
    m = len(final_predict)
    corr = 0.0
    for i in range(m):
        if data_train[i][-1] * final_predict[i] > 0:
            corr += 1
    return corr/m


def save_model(trees_result, trees_feature, result_file, feature_file):
    m = len(trees_feature)
    f_fea = open(feature_file, 'w')
    for i in range(m):
        fea_tmp = []
        for x in trees_feature[i]:
            fea_tmp.append(str(x))
        f_fea.writelines("\t".join(fea_tmp)+"\n")
    f_fea.close()

    with open(result_file, 'w') as f:
        pickle.dump(trees_result, f)


if __name__ == "__main__":
    data_train = load_data("../../../train_data/random_forest.train")
    trees_result, trees_feature = rdf.random_forest_training(data_train, 50)
    result = get_predict(trees_result, trees_feature, data_train)
    corr_rate = calc_correct_rate(data_train, result)
    print("\t-----correct rate:", corr_rate)
    save_model(trees_result, trees_feature, "../../../model/model.rdf.res", "../../../model/model.rdf.fea")

