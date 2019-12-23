from math import log

'''
决策树之ID3，基于信息增益进行特征选择
'''
class DecisionTreeID3:

    # 计算信息熵
    def calc_ent(self, data_set):
        '''
        :param data_set:数据集合
        :return: 信息熵
        '''
        num_entries = len(data_set)
        label_counts = {}
        for feat_vec in data_set:
            cur_label = feat_vec[-1]
            if cur_label not in label_counts.keys():
                label_counts[cur_label] = 0
            label_counts[cur_label] += 1

        ent = 0.0
        for label in label_counts.keys():
            prob = label_counts.get(label) / num_entries
            ent += -prob * log(prob, 2)
        return ent

    # 划分数据集
    def split_data_set(self, data_set, i, value):
        '''
        :param data_set: 数据集
        :param i: 第i个特征
        :param value: 第i个特征中的第value个值
        :return: 数据子集
        '''
        ret_data_set = []

        for feat_vec in data_set:
            # 假如属于相同的子集，则将当前特征的值切割
            if feat_vec[i] == value:
                reduced_feat_vec = feat_vec[:i]  # 不包含第i个值
                reduced_feat_vec.extend(feat_vec[i+1:])  # 从第i+1个值开始到最后
                ret_data_set.append(reduced_feat_vec)
        return ret_data_set

    def majority_cnt(self, class_list):
        '''
        :param class_list:分类的list
        :return: 数量最多的分类
        '''
        class_counts = {}
        for i in class_list:
            if i not in class_counts:
                class_counts[i] = 0
            class_counts[i] += 1
        return max(class_counts.items(), key=lambda x: x[1])[0]

    def choose_best_feature_to_split(self, data_set):
        # 特征数量
        n_feats = len(data_set[0]) - 1
        # 数据集整体信息熵
        base_ent = self.calc_ent(data_set)
        # 最优信息熵增益
        best_info_gain = 0.0
        best_feat = -1
        for i in range(n_feats):
            feat_list = [example[i] for example in data_set]
            unique_values = set(feat_list)
            cur_ent = 0
            # 计算当前特征下的信息熵
            for value in unique_values:
                sub_data_set = self.split_data_set(data_set, i, value)
                prob = len(sub_data_set) / (len(data_set) * 1.0)
                cur_ent += prob * self.calc_ent(sub_data_set)
            cur_info_gain = base_ent - cur_ent
            # 更新最优信息增益和最优特征
            if cur_info_gain > best_info_gain:
                best_info_gain = cur_info_gain
                best_feat = i
        return best_feat

    def create_tree(self, data_set, cols):
        '''
        :param data_set: 数据集合
        :param cols: 特征名称列表
        :return: ID3决策树
        '''
        class_list = [example[-1] for example in data_set]
        # 假如class_list只包含一个分类
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        # 如果数据维度为1，要基于最后一个特征划分，相当于接下来没有特征了，特征列表为空
        if len(data_set[0]) == 1:
            return self.majority_cnt(class_list)

        best_feat = self.choose_best_feature_to_split(data_set)
        best_feat_col = cols[best_feat]
        print(u"此时最优索引为： " + (best_feat_col))
        # 初始化决策树
        ID3Tree = {best_feat_col: {}}
        del (cols[best_feat])
        # 得到最优索引对应的属性值
        feat_values = [example[best_feat] for example in data_set]
        unique_values = set(feat_values)

        for value in unique_values:
            sub_cols = cols[:]
            sub_data_set = self.split_data_set(data_set, best_feat, value)
            ID3Tree[best_feat_col][value] = self.create_tree(sub_data_set, sub_cols)

        return ID3Tree
