class DecisionTreeCART:

    # 计算基尼系数
    def calc_gini(self, data_set):
        gini = 1.0
        label_counts = {}
        label_list = [example[-1] for example in data_set]
        for label in label_list:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

        for label in label_counts.keys():
            prob = label_counts[label] / (len(data_set) * 1.0)
            gini = gini - pow(prob, 2)
        return gini

    # 数据切分
    def split_data_set(self, data_set, feat, value):
        sub_data_set = []
        no_data_set = []
        for feat_vec in data_set:
            if feat_vec[feat] == value:
                reduced_feat_vec = feat_vec[:feat]
                reduced_feat_vec.extend(feat_vec[feat + 1:])
                sub_data_set.append(reduced_feat_vec)
            else:
                reduced_feat_vec = feat_vec[:feat]
                reduced_feat_vec.extend(feat_vec[feat + 1:])
                no_data_set.append(reduced_feat_vec)
        return sub_data_set, no_data_set

    # 选择最优特征和最优切分点
    def choose_best_feat_point_to_split(self, data_set):
        num_features = len(data_set[0]) - 1
        best_gini = 99999.0
        best_feat = -1
        best_value = -1

        for feat in range(num_features):
            feat_values = [example[feat] for example in data_set]
            unique_feat_values = set(feat_values)
            for value in unique_feat_values:
                sub_data_set, no_data_set = self.split_data_set(data_set, feat, value)
                prob = len(sub_data_set) / (len(data_set) * 1.0)
                no_prob = len(no_data_set) / (len(no_data_set) * 1.0)
                # 计算基尼系数
                cur_gini = prob * self.calc_gini(sub_data_set) + no_prob * self.calc_gini(no_data_set)

                # 更新基尼系数
                if cur_gini < best_gini:
                    best_gini = cur_gini
                    best_feat = feat
                    best_value = value
        return best_feat, best_value

    # 根据label数量进行投票
    def majority_vote(self, label_list):
        label_counts = {}

        for label in label_list:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        return max(label_counts.items(), lambda x: x[1])[0]

    # 创建树
    def create_tree(self, data_set, cols):
        label_list = [example[-1] for example in data_set]
        # 终止条件1，label类别全部一致
        if label_list.count(label_list[0]) == len(label_list):
            return label_list[0]
        # 终止条件2，没有特征可以继续分裂
        if len(data_set[0]) == 1:
            return self.majority_vote(data_set)

        # 寻找最优分裂特征和最优切分点
        best_feat, value = self.choose_best_feat_point_to_split(data_set)
        best_feat_col = cols[best_feat]

        # 构建cart树
        CARTTree = {best_feat_col: {}}
        del (cols[best_feat])

        sub_data_set, no_data_set = self.split_data_set(data_set, best_feat, value)
        cur_cols = cols[:]
        CARTTree[best_feat_col][value] = self.create_tree(sub_data_set, cur_cols)
        CARTTree[best_feat_col]['-1'] = self.create_tree(no_data_set, cur_cols)

        return CARTTree
