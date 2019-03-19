import numpy as np


def cos_sim(x, y):
    numerator = x * y.T
    denominator = np.sqrt(x * x.T) * np.sqrt(y * y.T)
    return (numerator / denominator)[0, 0]


def similarity(data):
    m = np.shape(data)[0]
    w = np.mat(np.zeros((m, m)))
    for i in range(m):
        for j in range(i, m):
            if j != 1:
                w[i, j] = cos_sim(data[i, ], data[j, ])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w


def user_bases_recommend(data, w, user):
    m, n = np.shape(data)
    interation = data[user, ]

    not_inter = []
    for i in range(n):
        if interation[0, i] == 0:
            not_inter.append(i)

    predict = {}
    for x in not_inter:
        item = np.copy(data[:, x])
        for i in range(m):
            if item[i, 0] != 0:
                if x not in predict:
                    predict[x] = w[user, i] * item[i, 0]
                else:
                    predict[x] = predict[x] + w[user, i] * item[i, 0]
    return sorted(predict.items(), key=lambda d:d[1], reverse=True)


def item_based_recommend(data, w, user):
    m, n = np.shape(data)
    interaction = data[:, user].T

    not_inter = []
    for i in range(n):
        if interaction[0, i] == 0:
            not_inter.append(i)

    predict = {}
    for x in not_inter:
        item = np.copy(interaction)
        for j in range(m):
            if item[0, j] != 0:
                if x not in predict:
                    predict[x] = w[x, j] * item[0, j]
                else:
                    predict[x] = predict[x] + w[x,j] * item[0, j]
    return sorted(predict.items(), key= lambda d:d[1], reverse=True)


def load_data(file_path):
    f = open(file_path)
    data = []
    for line in f.readlines():
        lines = line.strip().split('\t')
        tmp = []
        for x in lines:
            if x != '-':
                tmp.append(float(x))
            else:
                tmp.append(0)
        data.append(tmp)
    f.close()
    return np.mat(data)


def top_k(predict, k):
    top_recom = []
    len_result = len(predict)
    if k>= len_result:
        top_recom = predict
    else:
        top_recom = [predict[v] for v in range(k)]
    return top_recom


def main():
    data = load_data('../../../train_data/CF.train')
    w = similarity(data)
    predict = user_bases_recommend(data, w, 0)
    top_recom = top_k(predict, 2)
    print(top_recom)

if __name__ == '__main__':
    main()




