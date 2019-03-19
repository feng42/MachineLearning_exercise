import numpy as np
import random as rd

TRAIN_DIR = "../train_data/"
TEST_DIR = "../test_data/"
MODEL_DIR = "../model/"
RES_DIR = "../result/"
TRAIN_DATA = TRAIN_DIR + "softmax_regression.train"
#TEST_DATA = TEST_DIR + "softmax_regression.test"
MODEL_DATA = MODEL_DIR + "model.sr"
RES = RES_DIR + "res.sr"

TRAIN_STEPS = 1000
LEARNING_RATE = 0.1
NUM_TEST_SAMPLES = 4000

class softmax_regression_train(object):
    def __init__(self, TRAIN_DATA, MAX_STEPS, LEARNING_RATE, MODEL_DATA):
        feature, label, num_catagory = self.load_data(TRAIN_DATA)
        weights = self.gradientAscent(feature, label, num_catagory, MAX_STEPS, LEARNING_RATE)
        self.save_data(MODEL_DATA, weights)

    def gradientAscent(self, feature_data, label_data, num_category, max_steps, learning_rate):
        m, n = np.shape(feature_data)
        weights =np.mat(np.ones((n, num_category)))
        i = 0
        while i <= max_steps:
            err = np.exp(feature_data * weights)
            if i % 100 == 0:
                print("\t ---iter: ", i, ", cost: ",self.cost(err, label_data))
            rowsum = -err.sum(axis=1)
            rowsum = rowsum.repeat(num_category, axis=1)
            err /= rowsum
            for x in range(m):
                err[x, label_data[x, 0]] +=1
            weights += (learning_rate / m) * feature_data.T * err
            i += 1
        return weights

    def cost(self, err, label_data):
        m = np.shape(err)[0]
        sum_cost = 0.0
        for i in range(m):
            if err[i, label_data[i, 0]] / np.sum(err[i, :]) >0:
                sum_cost -= np.log(err[i, label_data[i, 0]] / np.sum(err[i, :]))
            else:
                sum_cost -= 0
        return sum_cost / m

    def load_data(self, file_name):
        with open(file_name, 'r') as fr:
            feature_data = []
            label_data = []
            for line in fr.readlines():
                feature_tmp = []
                #label_tmp = []
                lines =line.strip().split('\t')
                feature_tmp.append(1)
                for i in range(len(lines) - 1):
                    feature_tmp.append(float(lines[i]))
                label_data.append(int(lines[-1]))

                feature_data.append(feature_tmp)
                #label_data.append(label_tmp)
        return np.mat(feature_data), np.mat(label_data).T, len(set(label_data))

    def save_data(self, file_name, weights):
        with open(file_name, 'w') as fw:
            m, n = np.shape(weights)
            for i in range(m):
                w_tmp = []
                for j in range(n):
                    w_tmp.append(str(weights[i, j]))
                fw.write("\t".join(w_tmp)+"\n")
                
class softmax_regression_test(object):
    def __init__(self, MODEL_DATA, NUM_TEST_SAMPLES, RES):
        w, m, n = self.load_weights(MODEL_DATA)
        test_data = self.load_data(NUM_TEST_SAMPLES, m)
        result = self.predict(test_data, w)
        self.save_result(RES, result)

    def load_weights(self, file_name):
        with open(file_name, "r") as fr:
            w = []
            for line in fr.readlines():
                lines = line.strip().split('\t')
                w_tmp = []
                for x in lines:
                    w_tmp.append(float(x))
                w.append(w_tmp)
        weights = np.mat(w)
        m, n = np.shape(w)
        return weights, m, n

    def load_data(self, num, m):
        testDataSet = np.mat(np.ones((num, m)))
        for i in range(m):
            testDataSet[i, 1] = rd.random() * 6 - 3
            testDataSet[i, 2] = rd.random() * 15
        return testDataSet

    def predict(self, test_data, weights):
        h =test_data * weights
        return h.argmax(axis=1)

    def save_result(self, file_name, result):
        with open(file_name, 'w') as fw:
            m = np.shape(result)[0]
            for i in range(m):
                fw.write(str(result[i, 0]) + '\n')

if __name__ == "__main__":
    train = softmax_regression_train(TRAIN_DATA, TRAIN_STEPS, LEARNING_RATE, MODEL_DATA)
    test = softmax_regression_test(MODEL_DATA, NUM_TEST_SAMPLES, RES)
    