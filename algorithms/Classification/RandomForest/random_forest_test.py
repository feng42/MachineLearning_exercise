import cPickle as pickle
from random_forest import predict
from random_forest_train import get_predict

def load_data(file_name):
    f = open(file_name)
    test_data = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        tmp = []
        for x in lines:
            tmp.append(float(x))
        tmp.append(0)
        test_data.append(tmp)
    f.close()
    return  test_data

def load_model(result_file, feature_file):
    trees_fiture = []
    f_fea = open(feature_file)
    for line in f_fea.readlines():
        lines = line.strip().split("\t")
        tmp = []
        for x in lines:
            tmp.append(int(x))
        trees_fiture.append(tmp)
    f_fea.close()

    with open(result_file, 'r') as f:
        trees_result = pickle.load(f)

    return trees_result, trees_fiture


def save_result(data_test, prediction, result_file):
    m = len(prediction)
    n = len(data_test[0])

    f_result = open(result_file, 'w')
    for i in range(m):
        tmp = []
        for j in range(n-1):
            tmp.append(str(data_test[i][j]))
        tmp.append(str(prediction[i]))
        f_result.writelines("\t".join(tmp)+"\n")
    f_result.close()


if __name__ == "__main__":
    data_test = load_data("../test_data/random_forest.test")
    trees_result, trees_feature = load_model("../model/model.rdf.res", "../model/model.rdf.fea")
    prediction = get_predict(trees_result, trees_feature, data_test)
    save_result(data_test, prediction, "../result/res.rdf")