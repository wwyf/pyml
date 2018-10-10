import numpy as np
from pyml.emsemble.regression import GradientBoostingRegression
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
        2.5,5
    ])
    # rgs = GradientBoostingRegression()
    rgs = GradientBoostingRegression(learning_rate=0.2, base_estimator=DecisionTreeRegressor, max_tree_node_size=2, n_estimators=500)
    rgs.fit(mini_train_X,mini_train_Y)
    print(rgs.predict(mini_test_X))