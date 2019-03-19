import numpy as np
from random import normalvariate

TRAIN_DIR = "../train_data/"
TEST_DIR = "../test_data/"
MODEL_DIR = "../model/"
RES_DIR = "../result/"
TRAIN_DATA = TRAIN_DIR + "factorization_machine.train"
TEST_DATA = TEST_DIR + "factorization_machine.test"
MODEL_DATA = MODEL_DIR + "model.fm"
RES = RES_DIR + "res.fm"

VECTOR_DIMENTION = 3
MAX_STEPS = 10000
LEARNING_RATE = 0.01

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class factorization_machine_train(object):
    def __init__(self, TRAIN_DATA, v_dimension, max_iter, learning_rate, MODEL_DATA):
        dataTrain, labelTrain = self.loadDataSet(TRAIN_DATA)
        w0, w, v =self.stocGradAscent(np.mat(dataTrain), labelTrain, v_dimension, max_iter, learning_rate)
        predict_result = self.getPrediction(np.mat(dataTrain), w0, w, v)
        print("-----training error: %f" % (1- self.getAccuracy(predict_result, labelTrain)))
        self.save_model(MODEL_DATA, w0, w, v)

    def stocGradAscent(self, dataMatrix, classLabels, v_dimension, max_iters, learning_rate):
        m, n = np.shape(dataMatrix)
        w = np.zeros((n, 1))
        w0 = 0
        v = self.initialize_v(n, v_dimension)

        for it in range(max_iters):
            #print("iter: "+str(it))
            for x in range(m):
                inter_1 = dataMatrix[x] * v
                inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
                          np.multiply(v, v)
                interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.

                p = w0 + dataMatrix[x] *w + interaction
                loss = sigmoid((classLabels[x] * p[0, 0])) -1

                w0 = w0 - learning_rate * loss * classLabels[x]
                for i in range(n):
                    if dataMatrix[x, i] !=0:
                        w[i, 0] -= learning_rate * loss * classLabels[x] * dataMatrix[x, i]
                        for j in range(v_dimension):
                            v[i, j] -= learning_rate * loss * classLabels[x] * \
                                       (dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
            if it % 1000 == 0:
                print("\t --- iter: ", it, ", const: ", \
                    self.getCost(self.getPrediction(np.mat(dataMatrix), w0, w, v), classLabels))
        return w0, w, v

    def initialize_v(self, n, v_dimension):
        v = np.mat(np.zeros((n, v_dimension)))
        for i in range(n):
            for j in range(v_dimension):
                v[i, j] = normalvariate(0, 0.2)
        return v

    def getCost(self, predict, classLabels):
        m = len(predict)
        error =0.0
        for i in range(m):
            error -= np.log(sigmoid(predict[i] * classLabels[i]))
        return error

    def loadDataSet(self, file_name):
        dataMat = []
        labelMat = []
        with open(file_name, 'r') as fr:
            for line in fr.readlines():
                lines = line.strip().split('\t')
                lineArray = []
                for i in range(len(lines) -1):
                    lineArray.append(float(lines[i]))
                dataMat.append(lineArray)
                labelMat.append(float(lines[-1]) * 2 - 1)
        return dataMat, labelMat

    def getPrediction(self, dataMatrix, w0, w, v):
        m = np.shape(dataMatrix)[0]
        result = []
        for x in range(m):
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
                np.multiply(v, v)
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
            p = w0 +dataMatrix[x] * w + interaction
            pre = sigmoid(p[0, 0])
            result.append(pre)
        return result

    def getAccuracy(self, predict, classLabels):
        m = len(predict)
        allItem = 0
        error = 0
        for i in range(m):
            allItem += 1
            if float(predict[i]) < 0.5 and classLabels[i] == 1.0 :
                error += 1
            elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
                error += 1
            else:
                continue
        return float(error) / allItem

    def save_model(self, file_name, w0, w, v):
        with open(file_name, 'w') as fr:
            fr.write(str(w0)+'\n')
            w_array = []
            m = np.shape(w)[0]
            for i in range(m):
                w_array.append(str(w[i, 0]))
            fr.write("\t".join(w_array)+"\n")
            m1, n1 = np.shape(v)
            for i in range(m1):
                v_tmp = []
                for j in range(n1):
                    v_tmp.append(str(v[i, j]))
                fr.write("\t".join(v_tmp)+"\n")

class factorization_machine_test(object):
    def __init__(self, TEST_DATA, MODEL_DATA, RES):
        dataTest =self.loadDataSet(TEST_DATA)
        w0, w, v = self.loadModel(MODEL_DATA)
        result = self.getPrediction(dataTest, w0, w, v)
        self.save_result(RES, result)

    def loadDataSet(self, data):
        dataMat = []
        with open(data, 'r') as fr:
            for line in fr.readlines():
                lines = line.strip().split('\t')
                lineArray = []

                for i in range(len(lines)):
                    lineArray.append(float(lines[i]))
                dataMat.append(lineArray)
        return dataMat

    def loadModel(self, model_file):
        with open(model_file, 'r') as fr:
            line_index = 0
            w0 = 0.0
            w = []
            v= []
            for line in fr.readlines():
                lines = line.strip().split('\t')
                if line_index == 0:
                    w0 = float(lines[0].strip())
                elif line_index == 1:
                    for x in lines:
                        w.append(float(x.strip()))
                else:
                    v_tmp = []
                    for x in lines:
                        v_tmp.append(float(x.strip()))
                    v.append(v_tmp)
                line_index += 1
        return w0, np.mat(w).T, np.mat(v)

    def getPrediction(self, dataMatrix, w0, w, v):
        m = np.shape(dataMatrix)[0]
        result = []
        for x in range(m):
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
                np.multiply(v, v)
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
            p = w0 +dataMatrix[x] * w + interaction
            pre = sigmoid(p[0, 0])
            result.append(pre)
        return result

    def save_result(self, file_name, result):
        with open(file_name, 'w') as fw:
            fw.write('\n'.join(str(x) for x in result))

if __name__ == "__main__":
    train = factorization_machine_train(TRAIN_DATA, VECTOR_DIMENTION, MAX_STEPS, LEARNING_RATE, MODEL_DATA)
    test = factorization_machine_test(TEST_DATA, MODEL_DATA, RES)