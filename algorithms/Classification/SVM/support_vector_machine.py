import numpy as np



class SVM:
    def __init__(self, dataSet, labels, penality, toler, kernel_option):
        self.train_x = dataSet
        self.train_y = labels
        self.penal = penality
        self.toler =toler
        self.n_samples = np.shape(dataSet)[0]
        self.alphas = np.mat(np.zeros((self.n_samples, 1)))
        self.bias = 0
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2)))
        self.kernel_opt = kernel_option
        self.kernel_mat = calc_kernel(self.train_x, self.kernel_opt)


def calc_kernel(train_x, kernel_option):
    m = np.shape(train_x)[0]
    kernel_matrix = np.mat(np.zeros((m, m)))
    for i in range(m):
        kernel_matrix[:, i] = calc_kernel_value(train_x, train_x[i, :], kernel_option)
    return kernel_matrix


def calc_kernel_value(train_x, train_x_i, kernel_option):
    kernel_type = kernel_option[0]
    m = np.shape(train_x)[0]
    kernel_value = np.mat(np.zeros((m, 1)))
    if kernel_type == 'rbf':
        sigma = kernel_option[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(m):
            diff = train_x[i, :] - train_x_i
            kernel_value[i] = np.exp(diff * diff.T / (- 2.0 * sigma ** 2))
    else:
        kernel_value = train_x *train_x_i.T
    return kernel_value


def calc_error(svm, alpha_k):
    output_k = float(np.multiply(svm.alphas, svm.train_y).T \
                * svm.kernel_mat[:, alpha_k] + svm.bias)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k


def select_second_sample_j(svm, alpha_i, error_i):
    svm.error_tmp[alpha_i] = [1, error_i]
    candidateAlphaList = np.nonzero(svm.error_tmp[:, 0].A)[0]

    max_step = 0
    alpha_j = 0
    error_j = 0
    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i:
                continue
            error_k = calc_error(svm, alpha_k)
            if abs(error_k - error_i) > max_step:
                max_step = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(np.random.uniform(0, svm.n_samples))
        error_j = calc_error(svm, alpha_j)

    return alpha_j, error_j


def updata_error_tmp(svm, alpha_k):
    error = calc_error(svm, alpha_k)
    svm.error_tmp[alpha_k] = [1, error]


def choose_and_update(svm , alpha_i):
    error_i = calc_error(svm, alpha_i)
    if ((svm.train_y[alpha_i] * error_i < - svm.toler) and (svm.alphas[alpha_i] < svm.penal)) \
            or ((svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] >0)):

        alpha_j, error_j = select_second_sample_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.penal, svm.penal + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] +svm.alphas[alpha_i] - svm.penal)
            H = min(svm.penal, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            return 0

        eta = 2.0 * svm.kernel_mat[alpha_i, alpha_j] - svm.kernel_mat[alpha_i, alpha_i] -svm.kernel_mat[alpha_j, alpha_j]
        if eta >= 0:
            return 0

        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta

        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            updata_error_tmp(svm, alpha_j)
            return 0

        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
                               *(alpha_j_old - svm.alphas[alpha_j])

        b1 = svm.bias - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) * svm.kernel_mat[alpha_i, alpha_i] \
            - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) * svm.kernel_mat[alpha_i, alpha_j]
        b2 = svm.bias - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) * svm.kernel_mat[alpha_i, alpha_j] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) * svm.kernel_mat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.bias):
            svm.bias = b1
        elif (0 <svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.bias):
            svm.bias = b2
        else:
            svm.bias = (b1 + b2) / 2.0

        updata_error_tmp(svm, alpha_j)
        updata_error_tmp(svm, alpha_i)

        return 1
    else:
        return 0


def SVM_trainning(train_x, train_y, penality, toler, max_iter, kernel_option=('rbf', 0.431029)):
    svm = SVM(train_x, train_y, penality, toler, kernel_option)
    entireSet = True
    alpha_pairs_changed = 0
    iteration = 0

    while (iteration < max_iter) and ((alpha_pairs_changed > 0) or entireSet):
        print("\t iteration: ", iteration)
        alpha_pairs_changed = 0
        if entireSet:
            for x in range(svm.n_samples):
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1
        else:
            bound_samples = []
            for i in range(svm.n_samples):
                if 0 < svm.alphas[i, 0] < svm.penal:
                    bound_samples.append(i)
            for x in bound_samples:
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1

        if entireSet:
            entireSet = False
        elif alpha_pairs_changed == 0:
            entireSet = True

    return svm




