import numpy as np


class SuportVectorMachine:

    def __init__(self, max_iter=100, kernel='linear',C=1.0):
        self.max_iter = max_iter
        self.kernel = kernel
        self.C = C

    def init_args(self,X_train,y_train):
        self.X = X_train
        self.y = y_train
        self.m, self.n = X_train.shape
        # 初始化优化变量
        self.alpha = np.ones(self.m)
        self.b = 0.0

        # 初始化中间变量和松弛变量
        self.E = [self._E(i) for i in range(self.m)]


    def _compare(self, alpha, L, H):
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha

    def _eta(self, i1, i2):
        '''
        计算eta
        eta=k11+k22-2*k12
        :param i1:
        :param i2:
        :return:
        '''
        return self._kernel(self.X[i1], self.X[i1]) + self._kernel(self.X[i2], self.X[i2]) \
               - 2 * self._kernel(self.X[i1], self.X[i2])

    def _KKT(self, i):
        '''
        判定样本点是否满足kkt条件
        :param i:
        :return:
        '''
        yg = self._g(i) * self.y[i]
        if self.alpha[i] == 0:
            return yg >= 1
        elif 0 < self.alpha[i] < self.C:
            return yg == 1
        else:
            return yg <= 1

    def _select_alpha(self):
        '''
        选区符合条件的a1和a2
        外层循环，找出最不符合kkt条件的a1
        内层循环，根据a1的取值确定a2
        :return:
        '''
        # 找出在间隔边界上的样本点
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 其他样本点
        no_index_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(no_index_list)

        for i in index_list:
            if self._KKT(i):
                continue
            E1 = self.E[i]
            # 假如E1大于0，则取当E2最小的a2，否则取当E2最大的a2
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j

    def _kernel(self, x1, x2):
        '''
        核函数
        :param x1:
        :param x2:
        :return:
        '''
        if self.kernel == 'linear':
            return sum([x1[i] * x2[i] for i in range(self.n)])
        elif self.kernel == 'poly':
            return (sum([x1[i] * x2[i] for i in range(self.n)]) + 1) ** 2

    def _g(self, i):
        '''
        预测函数
        :param i:
        :return:
        '''
        b = self.b
        for j in range(self.m):
            b += self.alpha[j] * self.y[j] * self._kernel(self.X[i], self.X[j])
        return b

    def _E(self, i):
        '''
        g(x)预测值与真实值y的差值
        :param i:
        :return:
        '''
        return self._g(i) - self.y[i]

    def fit(self, X_train, y_train):
        '''
        :param X_train:
        :param y_train:
        :return:
        '''
        self.init_args(X_train,y_train)

        # 开始训练
        for epoch in range(self.max_iter):
            # 使用smo算法，首先选取两个变量alpha1和alpha2
            i1, i2 = self._select_alpha()

            # 确定a2的边界
            if self.y[i1] == self.y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]

            # 计算eta
            eta = self._eta(i1, i2)
            if eta <= 0:
                continue

            alpha2_new_unc = self.alpha[i2] + self.y[i2] * (E2 - E1) / eta
            alpha2_new = self._compare(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.y[i1] * self.y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.y[i1] * self._kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.y[i2] * self._kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.y[i1] * self._kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.y[i2] * self._kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
        return 'train done!'

    def predict(self, data):
        b = self.b
        for i in range(self.m):
            b += self.alpha[i] * self.y[i] * self._kernel(data, self.X[i])
        return 1 if b > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            res = self.predict(X_test[i])
            if res == y_test[i]:
                right_count += 1
        return right_count / len(X_test)

    def _weight(self):
        yx = self.y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w
