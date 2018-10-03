import numpy as np

from collections import Counter
def find_most_common(l):
    """
    寻找l列表中出现次数最多的元素
    """
    label_count = Counter(l)
    most_common_label = label_count.most_common(1)[0][0]
    return most_common_label

def entropy(l):
    """
    Parameters
    ----------
    l : 1d array-like shape(n_samples, )

    """
    feature_values, formatted_l = np.unique(l, return_inverse=True)
    counts = np.bincount(formatted_l)
    probabilities = counts/counts.sum()
    l_h = (-probabilities*np.log2(probabilities)).sum()
    return l_h

def condition_entropy(left, right):
    """
    Parameters
    ----------
    left : 1d array-like shape(n_samples, )
        条件对应的列
    right : 1d array-like shape(n_samples, )
        用来计算熵的一列 

    Returns
    ------------
    result : condition entropy
    """
    feature_values, formatted_l = np.unique(left, return_inverse=True)
    counts = np.bincount(formatted_l)
    probabilities = counts/counts.sum()
    entropies = np.zeros((len(feature_values)))
    for i,feature_value in enumerate(feature_values):
        # 对于每一个value
        # 先找出对应的y中的所有索引，取出相关的y
        # 然后调用计算熵的函数去计算
        this_indices = np.argwhere(left == feature_value).reshape(-1)
        entropies[i] = entropy(right[this_indices])
    result = (probabilities * entropies).sum()
    return result

def information_gain(left, right):
    """
    计算特征 left 对于 数据集right的信息增益

    Parameters
    ------------
    left : 1d array-like shape(n_samples, )
        条件对应的列
    right : 1d array-like shape(n_samples, )
        用来计算熵的一列 

    Returns
    ----------
    result : information_gain

    """
    return entropy(right) - condition_entropy(left, right)


def get_best_split_point(X, Y, method='id3', used_feature_list=[]):
    """寻找数据集X中最适合用于拆分的特征，并返回该特征索引

    Parameters
    ------------
    X : 2d array-like shape(n_samples, n_features)
    Y : 1d array-like shape(n_samples, )
    method : string
        - 'id3' use "information gain"

    Returns
    ---------
    split_point_index : the column index of split point

    """
    # # 先把已经用过的特征滤掉
    # used_feature_list = np.array(used_feature_list)
    # all_feature_list = np.arange(0,X.shape[1])
    # unused_feature_list = np.setdiff1d(all_feature_list,used_feature_list)
    # X = X[:,unused_feature_list]
    # print(X)
    # 每一个特征一个分数
    scores = np.zeros((X.shape[1]))
    # 遍历每一个特征，计算出一个分数
    for i in range(0, X.shape[1]):
        # 如果该列已经被用过了，不计算，直接当做0分 
        # TODO: 分数如果都是非负数的话那就没问题
        if  i in used_feature_list:
            scores[i] = 0
            continue
        # 取一列，该特征的所有值
        left = X[:,i].reshape(-1)
        right = Y
        if method == 'id3':
            scores[i] = information_gain(left, right)
        else :
            # TODO: 其他算法等待实现
            raise NotImplementedError
    # 取分数最高的那一个特征，并且返回这一个特征的索引
    return np.argmax(scores)

class DecisionTreeNode():
    def __init__(self):
        self.this_feature_index = None # 该节点对应的特征索引值
        self.children_node = {} # store the child nodes, access from feature value
        self.classification_result = None # label-like
        self.is_leaf = False

    def set_root_node(self, this_feature_index):
        self.this_feature_index = this_feature_index
    
    def set_leaf_node(self, classification_result):
        self.classification_result = classification_result
        self.if_leaf = True

    def get_next_node(self, x):
        """

        Parameters
        -----------
        x : 1d array-like shape(1, n_features)

        this_feature_value : the value of this feature

        Returns
        ---------
        next_node : the children node

        """
        this_feature_value = x[0][self.this_feature_index]
        if (self.is_leaf) :
            return None
        return self.children_node[this_feature_value]

    def get_classification_result(self, x):
        """
        Parameters
        ------------
        x : 1d array-like shape(1, n_features)

        Returns
        --------
        leaf_result
        """
        cur_node = self
        while (not cur_node.is_leaf):
            cur_node = cur_node.get_next_node(x)
        return cur_node.classification_result
    

# def build_tree(X, Y, method='id3', used_feature_list=[]):
#     """
#     Parameters
#     ------------
#     X : 2d array-like shape(n_samples, n_features)
#     Y : 1d array-like shape(n_samples, )

#     Returns
#     -----------
#     root_node : the root node of decision tree

#     """
#     # TODO: 深度信息如何用？如何判断应该停止建树？ 

#     # 1. 每一个特征都使用过了，此时应该停止建树
#     if len(used_feature_list) == X.shape[1]:
#         # 从Y中选择出现最多的那一个作为结果
#         classification_result = find_most_common(list(Y))
#         leaf_node = DecisionTreeNode()
#         leaf_node.set_leaf_node(classification_result)
#         return leaf_node
#     # 2. 需要建的这一个节点中的每一个样例都有相同的Y，即去重后只有1个Y,不需要再去拆分
#     if len(np.unique(Y)) == 1:
#         # 从Y中选择出现最多的那一个作为结果
#         classification_result = Y[0]
#         leaf_node = DecisionTreeNode()
#         leaf_node.set_leaf_node(classification_result)
#         return leaf_node

#     # 得到最佳拆分特征索引，以及更新用过了的特征的索引列表
#     split_point_index,used_feature_list = get_best_split_point(X,Y,method, used_feature_list)
#     # create the root node of Decision Tree
#     root_node = DecisionTreeNode(split_point_index)
#     # divide the dataset according split_point_index
#     feature_values, Xs, Ys = divide_dataset(X,Y,split_point_index)
#     for feature_value, X, Y in zip(feature_values, Xs, Ys):
#         root_node.children_node[feature_value] = build_tree(X,Y, used_feature_list)
#     return root_node



# class DecisionTreeClassifier():
#     """
#     寻找 向量[x1, x2, x3, ... , xn] -> 种类 y 的映射关系

#     Parameters
#     ----------
#     k : int, optional (default = 5)
#         Number of neighbors to use by default.

#     distance_func : callable, optional (default = euclidean_distance)
#         distance function used in find the neighbors

#         - [callable] : a function like f(vec1,vec2) 
#             which can return the distance(float).

#     Notes
#     ------
#     None
#     """

#     def fit(self, X, Y):
#         """

#         Parameters
#         -----------
#         X : array-like shape(n_samples, n_features)

#         Y : array-like shape(n_samples,) not-negative number

#         """

#     def predict(self, X_pred):
#         """

#         Parameters
#         -------------
#             X_pred : 2d array-like shape(n_samples, n_feature)

#         Returns
#         --------------
#             pre_Y : 1d array-like shape(n_samples,)

#         """