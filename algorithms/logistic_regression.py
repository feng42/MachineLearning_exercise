import numpy as np

TRAIN_DIR = "../train_data/"
TEST_DIR = "../test_data/"
MODEL_DIR = "../model/"
RES_DIR = "../result/"
TRAIN__DATA = TRAIN_DIR + "logistic_regression.train"
TEST_DATA = TEST_DIR + "logistic_regression.test"
MODEL_DATA = MODEL_DIR + "model.lr"
RES = RES_DIR + "res.lr"

TRAIN_STEPS = 1000
LEARNING_RATE = 0.1


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class logistic_regression_train(object):
    def __init__(self, TRAIN_DATA, MODEL, MAX_STEPS, LEARNING_RATE):
        self.feature, self.label = self.load_data(TRAIN_DATA)
        self.w =self.lr_train_bgd(self.feature, self.label, MAX_STEPS, LEARNING_RATE)
        self.save_data(MODEL, self.w)



    def lr_train_bgd(self, feature, label, max_steps, alpha):
        n = np.shape(feature)[1]
        w = np.mat(np.ones((n,1)))
        i = 0
        while i < max_steps:
            i += 1
            h = sigmoid(feature * w)
            err = label - h
            if i % 100 == 0:
                print("\t---iter= " + " , train error rate = "+str(self.error_rate(h, label)))
            w += alpha * feature.T * err
        return w

    def error_rate(self, h, label):
        m = np.shape(h)[0]
        sum_err = 0.0
        for i in range(m):
            if h[i, 0] > 0 and (1 - h[i, 0])>0 :
                sum_err -= (label[i,0] * np.log(h[i,0]) + (1 - label[i,0]) * np.log(1- h[i,0]))
            else:
                sum_err -= 0
        return sum_err

    def load_data(self, file_name):
        with open(file_name, 'r') as fr:
            feature_data = []
            label_data = []
            for line in fr.readlines():
                feature_tmp = []
                label_tmp = []
                lines =line.strip().split('\t')
                feature_tmp.append(1)
                for i in range(len(lines) - 1):
                    feature_tmp.append(float(lines[i]))
                label_tmp.append(float(lines[-1]))

                feature_data.append(feature_tmp)
                label_data.append(label_tmp)
        return np.mat(feature_data), np.mat(label_data)

    def save_data(self, file_name, w):
        m = np.shape(w)[0]
        with open(file_name, 'w') as fw:
            w_array = []
            for i in range(m):
                w_array.append(str(w[i, 0]))
            fw.write("\t".join(w_array))

class logistic_regression_test(object):
    def __init__(self, TEST_DATA, MODEL, RESULT):
        self.w = self.load_weights(MODEL)
        self.n = np.shape(self.w)[1]
        self.testData = self.load_data(TEST_DATA, self.n)
        self.h = self.predict(self.testData, self.w)
        self.save_result(RESULT, self.h)

    def load_weights(self, file_name):
        with open(file_name, "r") as fr:
            w = []
            for line in fr.readlines():
                lines = line.strip().split('\t')
                w_tmp = []
                for x in lines:
                    w_tmp.append(float(x))
                w.append(w_tmp)
            return np.mat(w)

    def load_data(self, file_name, n):
        with open(file_name, 'r') as fr:
            feature_data = []
            for line in fr.readlines():
                feature_tmp = []
                lines = line.strip().split('\t')
                if len(lines) != n - 1:
                    continue
                feature_tmp.append(1)
                for x in lines:
                    feature_tmp.append(float(x))
                feature_data.append(feature_tmp)
        return np.mat(feature_data)

    def predict(self, data, w):
        h = sigmoid(data * w.T)
        m = np.shape(h)[0]
        for i in range(m):
            if h[i, 0] < 0.5:
                h[i, 0] = 0.0
            else:
                h[i, 0] = 1.0
        return h

    def save_result(self, file_name, result):
        m = np.shape(result)[0]
        tmp = []
        for i in range(m):
            tmp.append(str(result[i, 0]))
        with open(file_name, 'w') as fw:
            fw.write("\t".join(tmp))

if __name__ == "__main__":
    train = logistic_regression_train(TRAIN__DATA, MODEL_DATA, TRAIN_STEPS, LEARNING_RATE)
    test = logistic_regression_test(TEST_DATA, MODEL_DATA, RES)