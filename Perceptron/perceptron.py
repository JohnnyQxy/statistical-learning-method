import numpy as np


class Perceptron:

    def __init__(self, eta=0.1, max_iter=10000):
        self.coef_ = None
        self.intercept_ = None
        # self._thetas=None
        self.eta = eta
        self.max_iter = max_iter

    def df_loss(self, x, y):
        return np.dot(y, x), y

    def loss_fun(self, x, y):
        return -y * (np.dot(self.coef_, x) + self.intercept_)

    def fit(self, X_train, y_train):

        self.m, self.n = X_train.shape
        # 初始化参数
        self.coef_ = np.ones(self.n)
        self.intercept_ = 0.0

        for epoch in range(self.max_iter):
            wrong_num = 0
            for i in range(self.m):
                if self.loss_fun(X_train[i], y_train[i]) >= 0:
                    wrong_num += 1
                    df_coef, df_intercept = self.df_loss(X_train[i], y_train[i])
                    self.coef_ = self.coef_ + self.eta * df_coef
                    self.intercept_ = self.intercept_ + self.eta * df_intercept

            if wrong_num == 0:
                break

        return 'train done!'
