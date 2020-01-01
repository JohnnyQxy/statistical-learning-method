import numpy as np


class AdaBoost:

    def __init__(self, classifiers_num=50, learning_rate=1.0):
        '''
        初始化函数
        :param classifiers_num: 基分类器个数
        :param learning_rate: 学习速率
        '''

        self.classifier_num = classifiers_num
        self.learning_rate = learning_rate

        # 弱分类器集合
        self._base_classifiers = []

        # 弱分类器的权重
        self._alpha = []

    def create_base_classifier(self, data, labels, weights):
        '''
        构建基本分类器
        :param data: 数据集
        :param labels: 标签
        :param weights: 数据集权重
        :return:
        '''
        M, N = data.shape

        best_error = 100000.0
        best_feature = -1
        best_seg_point = -1
        best_direct = None
        best_array = None

        for feature in range(N):

            feature_values = data[:, feature]

            values_len = len(feature_values)
            # 求切分点
            max_value = max(feature_values)
            min_value = min(feature_values)

            seg_points_num = (max_value - min_value) // self.learning_rate
            seg_points = [min_value + self.learning_rate * i for i in range(1, int(seg_points_num) + 1)]

            for seg_point in seg_points:

                if seg_point not in feature_values:
                    cp_array_pos = np.array([1 if feature_values[i] > seg_point else -1 for i in range(values_len)])
                    # 求分类误差率
                    weights_error_pos = np.sum([weights[i] for i in range(values_len) if cp_array_pos[i] != labels[i]])

                    cp_array_neg = np.array([-1 if feature_values[i] > seg_point else 1 for i in range(values_len)])
                    # 求分类误差率
                    weights_error_neg = np.sum([weights[i] for i in range(values_len) if cp_array_neg[i] != labels[i]])

                    if weights_error_pos < weights_error_neg:
                        weights_error = weights_error_pos
                        cp_array = cp_array_pos
                        direct = 'positive'
                    else:
                        weights_error = weights_error_neg
                        cp_array = cp_array_neg
                        direct = 'negtive'
                    if weights_error < best_error:
                        best_error = weights_error
                        best_feature = feature
                        best_seg_point = seg_point
                        best_direct = direct
                        best_array = cp_array

        return best_error, best_feature, best_seg_point, best_direct, best_array

    def calc_alpha(self, em):
        '''
        求基分类器权重
        :param em: 误差率
        :return:
        '''
        return 0.5 * np.log((1 - em) / em)

    def calc_z(self, weights, alpha, labels, cp_array):
        '''
        求规范因子
        :param weights:训练数据集权重
        :param alpha:基分类器权重
        :param labels:数据集标签
        :param cp_array:基分类器预测集
        :return:
        '''
        return np.sum([weights[i] * np.exp(-alpha * labels[i] * cp_array[i]) for i in range(len(labels))])

    def update_weights(self, weights, z, alpha, labels, cp_array):
        '''
        更新训练数据集权重
        :param weights:
        :param z:
        :param alpha:
        :param labels:
        :param cp_array:
        :return:
        '''
        return [weights[i] * np.exp(-alpha * labels[i] * cp_array[i]) / z for i in range(len(labels))]

    def fit(self, X_train, y_train):
        '''
        训练模型
        :param X_train: 特征值
        :param y_train: 标签值
        :return:
        '''
        # 初始化训练集权重
        M, N = X_train.shape
        weights = [1 / M] * M
        # 开始训练
        for epoch in range(self.classifier_num):

            # 构建基本分类器
            error, feature, seg_point, direct, cp_array = self.create_base_classifier(X_train, y_train, weights)

            if error == 0:
                break

            # 求此基本分类器的次数alpha
            alpha = self.calc_alpha(error)
            self._alpha.append(alpha)

            # 添加基本分类器参数
            self._base_classifiers.append((feature, seg_point, direct))

            # 计算规范化因子
            z = self.calc_z(weights, alpha, y_train, cp_array)
            # 更新训练集权重分布
            weights = self.update_weights(weights, z, alpha, y_train, cp_array)

            print('classifier:{}/{} error:{:.3f} seg_point:{} direct:{} alpha:{:.5f}'.format(epoch + 1,
                                                                                             self.classifier_num, error,
                                                                                             seg_point, direct, alpha))
            print('weights:{}'.format(weights))
            print('\n')

    def calc_g(self, value, seg_point, direct):
        if direct == 'positive':
            return 1 if value > seg_point else -1
        else:
            return -1 if value > seg_point else 1

    def predict(self, values):
        result = 0.0
        for i in range(len(self._base_classifiers)):
            feature, seg_point, direct = self._base_classifiers[i]
            value = values[feature]
            result += self._alpha[i] * self.calc_g(value, seg_point, direct)
        return result

    def score(self, X_test, y_test):
        '''
        打分函数
        :param X_test:
        :param y_test:
        :return:
        '''
        right_num = 0
        for i in range(len(X_test)):
            values = X_test[i]
            if self.predict(values) == y_test[i]:
                right_num += 1

        return right_num / len(X_test)
