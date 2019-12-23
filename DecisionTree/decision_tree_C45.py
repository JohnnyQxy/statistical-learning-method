from math import log

'''
决策树之C4.5,基于信息增益比进行特征选择
'''

class DecisionTreeC45:

    #计算信息熵
    def calc_ent(self,data_set):


    def choose_best_feature_to_split(self,data_set):
        base_ent=calc_ent(data_set)

    def majority_cnt(self,label_list):
        label_counts={}
        for i in label_list:
            if i not in label_counts.keys():
                label_counts[i]=0
            label_counts[i]+=1
        return max(label_counts.items(),lambda x:x[1])[0]

    def create_tree(self,data_set,cols):
        label_list=[example[-1] for example in data_set]
        #假如label只有一种
        if label_list.count(label_list[0])==len(label_list):
            return label_list[0]
        #假如只剩下label列
        if len(data_set[0]==1):
            return self.majority_cnt(label_list)

        best_feat=choose_best_feature_to_split(data_set)
        best_feat_col=cols[best_feat]
        return None