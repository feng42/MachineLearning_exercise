import cPickle as pickle
import numpy as np
from support_vector_machine import *
from main_svm_train import svm_predict

def load_test_data(test_file):
    data = []
    f = open(test_file)
    for line in f.readlines():
        lines = line.strip().split(' ')
        index = 0
        tmp = []
        for i in range(0, len(lines)):
            li = lines[i].strip().split(':')
            if int(li[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                while(int(li[0]) - 1 > index):
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp)<13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.mat(data)


def load_svm_model(svm_model_file):
    with open(svm_model_file, 'r') as fr:
        svm_model = pickle.load(fr)
    return svm_model


def get_prediction(test_data, svm):
    m = np.shape(test_data)[0]
    prediction = []
    for i in range(m):
        predict = svm_predict(svm, test_data[i, :])
        prediction.append(str(np.sign(predict)[0,0]))
    return prediction


def save_prediction(result_file, prediction):
    f = open(result_file, 'w')
    f.write(' '.join(prediction))
    f.close()

if __name__ == "__main__":
    test_data = load_test_data("../test_data/svm.test")
    svm_model = load_svm_model("../model/model.svm")
    prediction = get_prediction(test_data, svm_model)
    save_prediction("../result/res.svm", prediction)