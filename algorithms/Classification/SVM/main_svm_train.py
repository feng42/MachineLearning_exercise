#coding:UTF-8
import numpy as np
#import support_vector_machine as svm
from algorithms.support_vector_machine import *
import cPickle as pickle

def load_data_libsvm(data_file):
    data = []
    label = []
    with open(data_file, 'r') as fr:
        for line in fr.readlines():
            lines = line.strip().split(' ')
            label.append(float(lines[0]))
            index = 0
            tmp = []
            for i in range(1, len(lines)):
                li = lines[i].strip().split(":")
                if int(li[0]) - 1 == index:
                    tmp.append(float(li[1]))
                else:
                    while int(li[0]) - 1 > index:
                        tmp.append(0)
                        index += 1
                    tmp.append(float(li[1]))
                index += 1
            while len(tmp) < 13:
                tmp.append(0)
            data.append(tmp)
    return np.mat(data), np.mat(label).T


def svm_predict(svm, test_smple_x):
    kernel_value = calc_kernel_value(svm.train_x, test_smple_x, svm.kernel_opt)
    predict = kernel_value.T * np.multiply(svm.train_y, svm.alphas) + svm.bias
    return predict


def calc_accuracy(svm, test_x, test_y):
    n_samples = np.shape(test_x)[0]
    correct = 0.0
    for i in range(n_samples):
        predict = svm_predict(svm, test_x[i, :])
        if np.sign(predict) == np.sign(test_y[i]):
            correct += 1
    accuracy = correct / n_samples
    return accuracy


def save_svm_model(svm_model, model_file):
    with open(model_file, 'w') as fw:
        pickle.dump(svm_model, fw)


if __name__ == "__main__":
    dataset, labels = load_data_libsvm("../train_data/svm.train")
    penal = 0.6
    toler = 0.001
    maxIter = 500
    svm_model = SVM_trainning(dataset, labels, penal, toler, maxIter)
    accuracy = calc_accuracy(svm_model, dataset, labels)
    print("The Trainning accuracy is: %.3f%%" %(accuracy*100))
    save_svm_model(svm_model, "../model/model.svm")