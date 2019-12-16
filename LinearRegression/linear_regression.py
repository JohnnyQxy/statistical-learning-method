import numpy as np

'''
线性回归
'''
class LinearRegression:

    # 初始化 Linear Regression模型
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    # 通过正规方程
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    # 梯度下降法求解参数
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        '''
        :param X_train: 训练集
        :param y_train: 标签
        :param eta: 学习率
        :param n_iters: 最大迭代次数
        :return:
        '''
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 定义损失函数
        def loss_fun(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(2 * y)
            except:
                return float('inf')

        # 对损失函数求导
        def df_loss_fun(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) / len(y)

        def gradient_descent(X_b, y, init_theta, eta=0.01, n_iters=1e4, epsilon=1e-9):
            '''
            :param X_b: 输入特征向量
            :param y: label
            :param init_theta: 初始theta值
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
                if (abs(loss_fun(theta, X_b, y) - loss_fun(last_theta, X_b, y)) < epsilon):
                    break
                cur_iter = cur_iter + 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        intial_theta = np.zeros(X_b.shape[1])

        self._theta = gradient_descent(X_b, y_train, init_theta=intial_theta)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    # 预测函数
    def predict(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of input must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    # 打分函数
    def score(self,X_test,y_test):
        assert X_test.shape[0]==y_test.shape[0],\
            "the size of X_test must be equal to the size of y_test"
        y_predict=self.predict(X_test)
        return self.r2_score(y_test,y_predict)

    # 评估函数
    def r2_score(self,y_true,y_predict):
        MSE=np.sum(((y_true-y_predict)**2)/len(y_true))
        return 1-MSE/np.var(y_true)