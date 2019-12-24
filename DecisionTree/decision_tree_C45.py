from math import log

'''
决策树之C4.5,基于信息增益比进行特征选择
'''


class DecisionTreeC45:

    # 计算信息熵
    def calc_ent(self, data_set):
        '''
        :param data_set: 数据集
        :return: 信息熵
        '''
        label_counts = {}
        # 统计各个label的数量
        for data_vec in data_set:
            label = data_vec[-1]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        # 计算经验熵
        ent = 0.0
        for label in label_counts.keys():
            prob = label_counts[label] / (len(data_set) * 1.0)
            ent += -prob * log(prob, 2)
        return ent

    # 切分数据集
    def split_data_set(self, data_set, i, value):
        '''
        :param data_set:
        :param i:
        :param value:
        :return:
        '''
        ret_data_set = []
        for feat_vec in data_set:
            if feat_vec[i] == value:
                reduced_data_vec = feat_vec[:i]
                reduced_data_vec.extend(feat_vec[i + 1:])
                ret_data_set.append(reduced_data_vec)
        return ret_data_set

    # 选择信息增益比最高的特征
    def choose_best_feature_to_split(self, data_set):
        '''
        :param data_set:
        :return:
        '''
        base_ent = self.calc_ent(data_set)
        num_feats = len(data_set[0]) - 1
        best_info_gain_ratio = 0.0
        best_feat = -1

        for i in range(num_feats):
            value_list = [example[i] for example in data_set]
            unique_value = set(value_list)
            cur_ent = 0.0
            for value in unique_value:
                sub_data_set = self.split_data_set(data_set, i, value)
                prob = len(sub_data_set) / (len(data_set) * 1.0)
                cur_ent += prob * self.calc_ent(sub_data_set)
            cur_info_gain_ratio = (base_ent - cur_ent) / base_ent
            # 更新最优信息增益比和最优特征
            if cur_info_gain_ratio > best_info_gain_ratio:
                best_info_gain_ratio = cur_info_gain_ratio
                best_feat = i
        return best_feat

    # 根据多数投票原则确定叶子节点结果
    def majority_cnt(self, label_list):
        '''
        :param label_list:
        :return:
        '''
        label_counts = {}
        for i in label_list:
            if i not in label_counts.keys():
                label_counts[i] = 0
            label_counts[i] += 1
        return max(label_counts.items(), lambda x: x[1])[0]

    # 构建决策树
    def create_tree(self, data_set, cols):
        '''
        :param data_set:
        :param cols:
        :return:
        '''
        label_list = [example[-1] for example in data_set]
        # 假如label只有一种
        if label_list.count(label_list[0]) == len(label_list):
            return label_list[0]
        # 假如只剩下label列
        if len(data_set[0]) == 1:
            return self.majority_cnt(label_list)

        # 选择最优特征
        best_feat = self.choose_best_feature_to_split(data_set)
        best_feat_col = cols[best_feat]
        print(u"此时最优索引为： " + (best_feat_col))
        # 构建决策树
        C45Tree = {best_feat_col: {}}
        del (cols[best_feat])

        best_feat_values = [example[best_feat] for example in data_set]
        unique_best_feat_values = set(best_feat_values)
        for value in unique_best_feat_values:
            sub_cols = cols[:]
            sub_data_set = self.split_data_set(data_set, best_feat, value)
            C45Tree[best_feat_col][value] = self.create_tree(sub_data_set, sub_cols)
        return C45Tree
