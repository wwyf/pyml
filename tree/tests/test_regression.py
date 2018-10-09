import numpy as np
from pyml.tree.regression import DecisionTreeRegressor

if __name__ == '__main__':
    mini_train_X = np.array([
        [1,2,3,4,5,6,7,8],
        [2,3,4,5,6,7,8,9],
        [3,4,5,6,7,8,9,10],
        [4,5,6,7,8,9,10,11],
        [5,6,7,8,9,10,11,12],
        [6,7,8,9,10,11,12,13],
        [7,8,9,10,11,12,13,14]
    ])
    mini_train_Y = np.array([
        1.5,2.5,3.5,4.5,5.5,6.5,7.5
    ])
    mini_test_X = np.array([
        [2,3,4,5,6,7.5,8,9],
        [4,5,6,7.5,8,9,10,11]
    ])
    mini_standard_out_Y = np.array([
        2.5,4.5
    ])
    rgs = DecisionTreeRegressor()
    rgs.fit(mini_train_X,mini_train_Y)
    print(rgs.predict(mini_test_X))
    rgs.root_node.print_tree('test.png')