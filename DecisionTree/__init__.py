import DecisionTree.treePlotter as treePlotter
from DecisionTree.decision_tree_ID3 import DecisionTreeID3
from DecisionTree.decision_tree_C45 import DecisionTreeC45


def read_data(filename):
    '''
    :param filename:
    :return:
    '''
    fr = open(filename, 'r')
    all_lines = fr.readlines()
    labels = ['年龄段', '有工作', '有自己的房子', '信贷情况']
    data_set = []
    for line in all_lines[0:]:
        line = line.strip().split(',')  # 以逗号进行切分
        data_set.append(line)
    return data_set, labels


if __name__ == '__main__':
    # 验证ID3决策树
    filename = './data/dataset.txt'
    data_set, labels = read_data(filename)
    labels_tmp = labels[:]
    ID3Tree = DecisionTreeID3()
    ID3_tree = ID3Tree.create_tree(data_set, labels_tmp)
    treePlotter.ID3_Tree(ID3_tree)

    labels_tmp = labels[:]
    C45Tree = DecisionTreeC45()
    C45_tree = C45Tree.create_tree(data_set, labels_tmp)
    treePlotter.C45_Tree(C45_tree)
