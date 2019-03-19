import numpy as np


def ridge_regression(feature, label, lam):
    n = np.shape(feature)[1]
    w = (feature.T * feature + lam * np.mat(np.eye(n)).I * feature.T * label )
    return w


def get_error(feature, label, w):
    m = np.shape(feature)[0]
    left = (label - feature * w).T * (label - feature * w)
    return (left / (2 * m))[0, 0]


def get_gradient(feature, label, w, lam):
    err = (label - feature * w).T
    left = err * (-1) * feature
    return left.T + lam * w


def get_result(feature, label, w, lam):
    left = (label - feature * w).T * (label - feature * w)
    right = lam * w.T * w
    return (left + right)/2


def bfgs(feature, label, lam, maxCycle):
    n = np.shape(feature)[1]
    w0 = np.mat(np.zeros((n, 1)))
    rho = 0.55
    sigma = 0.4
    Bk = np.eye(n)
    k = 1

    while k < maxCycle:
        print('\titer: ',k,'\terror: ', get_error(feature, label, w0))
        gk = get_gradient(feature, label, w0, lam)
        dk = np.mat(-np.linalg.solve(Bk, gk))
        m = 0
        mk = 0
        while m < 20:
            newf = get_result(feature, label, (w0 + (rho ** m) * dk), lam)
            oldf = get_result(feature, label, w0, lam)
            #print(type(newf), type(oldf), np.shape(newf), np.shape(oldf))
            if (newf < oldf + sigma * (rho ** m) * (gk.T * dk)[0, 0]):
                mk = m
                break
            m += 1

        # BFGS
        w = w0 + rho ** mk * dk
        sk = w - w0
        yk = get_gradient(feature, label, w, lam) - gk
        if (yk.T * sk > 0):
            Bk = Bk - (Bk * sk * sk.T * Bk) / (sk.T * Bk * sk) + (yk * yk.T) / (yk.T * sk)
        k += 1
        w0 = w
    return w0


def lbfgs(feature, label, lam, maxCycle, m=10):
    n = np.shape(feature)[1]
    w0 = np.mat(np.zeros((n, 1)))
    rho = 0.55
    sigma = 0.4

    H0 = np.eye(n)

    s = []
    y = []

    k = 1
    gk = get_gradient(feature, label, w0, lam)
    dk = - H0 * gk
    while k < maxCycle:
        print('\titer: ', k, '\terror: ', get_error(feature, label, w0))
        m1 = 0
        mk = 0
        gk = get_gradient(feature, label, w0, lam)
        # Amijo
        while m1 < 20:
            newf = get_result(feature, label, (w0 + rho ** m1 * dk), lam)
            oldf = get_result(feature, label, w0, lam)
            if newf < oldf + sigma * (rho ** m1) * (gk.T * dk)[0, 0]:
                mk = m1
                break
            m1 += 1
        # LBFGS
        w = w0 + rho ** mk * dk
        if k > m:
            s.pop(0)
            y.pop(0)
        sk = w - w0
        qk = get_gradient(feature, label, w, lam)
        yk = qk - gk
        s.append(sk)
        y.append(yk)
        # two - loop
        t = len(s)
        a = []
        for i in range(t):
            alpha = (s[t-i-1].T * qk) / (y[t-i-1].T * s[t-i-1])
            qk = qk - alpha[0, 0] * y[t-i-1]
            a.append(alpha[0, 0])
        r = H0 * qk

        for i in range(t):
            beta = (y[i].T * r) / (y[i].T * s[i])
            r = r + s[i] * (a[t-i-1] - beta[0, 0])

        if yk.T * sk > 0:
            dk -= r
        k += 1
        w0 = w
    return w0


