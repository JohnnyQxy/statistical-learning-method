import numpy as np


class NativeBayes:

    def __init__(self):
        self.prior = {}
        self.model = {}

    def calc_model(self, data):
        '''
        计算p(x|y)的概率
        :return:
        '''
        model = {}
        for i, data in enumerate(zip(*data)):
            feat_dict = {}
            for j in data:
                if j not in feat_dict:
                    feat_dict[j] = 0
                feat_dict[j] += 1
            for key in feat_dict.keys():
                feat_dict[key] = feat_dict[key] / len(data)
            model[i] = feat_dict
        return model

    def calc_prior(self, labels):
        '''
        计算先验概率p(y)
        :param labels:
        :return:
        '''
        prior_dict = dict()
        for i in labels:
            if i not in prior_dict:
                prior_dict[i] = 0
            prior_dict[i] += 1

        # for key in prior_dict.keys():
        #     prior_dict[key] = prior_dict[key] / float(len(labels))

        return prior_dict

    def fit(self, train_X, train_y):
        '''
        模型训练
        :param train_X:
        :param train_y:
        :return:
        '''

        # 计算先验概率
        self.prior = self.calc_prior(train_y)

        labels = list(set(train_y))
        data = {label: [] for label in labels}
        for x, label in zip(train_X, train_y):
            data[label].append(x)
        self.model = {label: self.calc_model(value) for label, value in data.items()}

    def predict(self, test_x):
        pred = {}
        for label, value in self.model.items():
            pred[label] = self.prior[label]
            for cols, data in value.items():
                if test_x[cols] in data.keys():
                    pred[label] *= data[test_x[cols]]

        return sorted(pred.items(), key=lambda x: x[-1])[-1][0]

    def score(self, test_X, test_y):
        right_num = 0
        for x, y in zip(test_X, test_y):
            pred_y = self.predict(x)
            if pred_y == y:
                right_num += 1
        print('right count num is: ',right_num)
        return right_num / float(len(test_y))
