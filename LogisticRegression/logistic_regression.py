import numpy as np

'''
逻辑回归
'''


class LogisticRegression:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    # 定义激励函数
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    # 梯度下降法求解
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        '''
        :param X_train:
        :param y_train:
        :param eta: 步长
        :param n_iters: 最大迭代次数
        :return:
        '''
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 定义损失函数
        def loss_fun(theta, X_b, y):
            fx = self.sigmoid(X_b.dot(theta))
            m, n = np.shape(X_b)
            try:
                return np.sum(-y.T.dot(np.log(fx)) - (np.ones((m, 1)) - y.T).dot(np.log(1 - fx))) / len(y)
            except:
                return float('inf')

        # 对损失函数求导
        def df_loss_fun(theta, X_b, y):
            fx = self.sigmoid(X_b.dot(theta))
            return X_b.T.dot(fx - y) / len(y)

        # 梯度下降过程
        def gradient_decent(X_b, y, init_theta, eta=0.01, n_iters=10000, epsilon=1e-7):
            '''
            :param X_b: 输入特征向量
            :param y: 标签
            :param init_theta: 初始theta
            :param eta: 步长
            :param n_iters: 最大迭代次数
            :param epsilon: 容忍度
            :return:
            '''

            theta = init_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df_loss_fun(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(loss_fun(theta, X_b, y) - loss_fun(last_theta, X_b, y))<epsilon):
                    break
                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        init_theta = np.ones(X_b.shape[1])

        self._theta = gradient_decent(X_b, y_train, init_theta=init_theta)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

    # 预测函数
    def predict(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of input must be equal to X_train!"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self.sigmoid(X_b.dot(self._theta))

    # 评估函数
    def score(self, X_test, y_test):
        assert X_test.shape[0] == y_test.shape[0], \
            "the size of X_test must be equal to the size of y_test"
        y_predict = self.predict(X_test)
        return self.r2_score(y_test, y_predict)

    def r2_score(self, y_true, y_predict):
        MSE = np.sum((y_true - y_predict) ** 2 / len(y_true))
        return 1 - MSE / np.var(y_true)
