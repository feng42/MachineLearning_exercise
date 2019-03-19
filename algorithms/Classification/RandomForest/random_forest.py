import numpy as np
import random as rd
from math import log


def cal_gini_index(data):
    total_sample = len(data)
    if len(data) == 0:
        return 0
    label_counts = label_uniq_cnt(data)

    gini = 0
    for label in label_counts:
        gini += pow(label_counts[label], 2)
    gini = 1 - float(gini) / pow(total_sample, 2)
    return gini


def label_uniq_cnt(data):
    label_uniq_count = {}
    for x in data:
        label = x[len(x) - 1]
        #print(x)
        #print(type(x),type(label))
        if label not in label_uniq_count:
            label_uniq_count[label] = 0
        label_uniq_count[label] = label_uniq_count[label] + 1
    return label_uniq_count


class Node(object):
    def __init__(self, fea=-1, value=None, results=None, right=None, left=None):
        self.fea = fea
        self.value = value
        self.results = results
        self.right = right
        self.left = left


def build_tree(data):
    if len(data) == 0:
        return Node()

    currentGini = cal_gini_index(data)

    bestGain = 0.0
    bestCriteria = None
    bestSets = None

    feature_num = len(data[0]) - 1
    for fea in range(0, feature_num):
        feature_values = {}
        for sample in data:
            feature_values[sample[fea]] = 1

        for value in feature_values.keys():
            set_1, set_2 = split_tree(data, fea, value)
            nowGini = float(len(set_1) * cal_gini_index(set_1) +\
                            len(set_2) * cal_gini_index(set_2)) / len(data)
            gain = currentGini - nowGini
            if gain>bestGain and len(set_1) > 0 and len(set_2) > 0:
                bestGain = gain
                bestCriteria = (fea,value)
                bestSets = (set_1, set_2)
    if bestGain>0:
        right = build_tree(bestSets[0])
        left = build_tree(bestSets[1])
        return Node(fea=bestCriteria[0], value=bestCriteria[1], right=right, left=left)
    else:
        return Node(results=label_uniq_cnt(data))


def split_tree(data, fea, value):
    set_1 = []
    set_2 = []
    for x in data:
        if x[fea] >= value:
            set_1.append(x)
        else:
            set_2.append(x)
    return set_1, set_2


def predict(sample, tree):
    if tree.results != None:
        return tree.results
    else:
        val_sample= sample[tree.fea]
        branch = None
        if val_sample >= tree.value:
            branch = tree.right
        else:
            branch = tree.left
    return predict(sample, branch)


def choose_sample(data, k):
    m, n = np.shape(data)
    feature = []
    for j in range(k):
        feature.append(rd.randint(0, n-2))
    index = []
    for i in range(m):
        index.append(rd.randint(0, m - 1))
    data_samples = []
    for i in range(m):
        data_tmp = []
        for fea in feature:
            data_tmp.append(data[index[i]][fea])
        data_tmp.append(data[index[i]][-1])
        data_samples.append(data_tmp)
    return data_samples, feature


def random_forest_training(data_train, trees_num):
    trees_result = []
    trees_feature = []
    n = np.shape(data_train)[1]
    if n > 2:
        k = int(log(n - 1, 2)) + 1
    else:
        k = 1
    for i in range(trees_num):
        data_samples, feature = choose_sample(data_train, k)
        tree = build_tree(data_samples)
        trees_result.append(tree)
        trees_feature.append(feature)
    return trees_result, trees_feature