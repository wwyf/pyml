import unittest
import numpy as np

from pyml.tree.classification import entropy
from pyml.tree.classification import condition_entropy
from pyml.tree.classification import information_gain
from pyml.tree.classification import get_best_split_point
# from pyml.tree.classification import build_tree

class test_classification(unittest.TestCase):

    def test_entropy(self):
        test_data = np.array(['a','a','a','sdfsd','sdfsd','we'])
        test_result = entropy(test_data)
        self.assertAlmostEqual(test_result, 1.4591479)
    def test_condition_entropy(self):
        left = np.array(['y','y','m','o','o','o','m','y','y','o','y','m','m','o'])
        right = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
        test_result = condition_entropy(left, right)
        self.assertAlmostEqual(test_result, 0.6935361)
    def test_information_gain(self):
        left = np.array(['y','y','m','o','o','o','m','y','y','o','y','m','m','o'])
        right = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
        test_result = information_gain(left, right)
        self.assertAlmostEqual(test_result, 0.24674981)
    def test_get_best_split_point_id3(self):
        X = np.array([
            ['y','y','m','o','o','o','m','y','y','o','y','m','m','o'], 
            # age 0.246
            [0,0,0,0,1,1,1,0,1,1,1,0,1,0], 
            # is student? 0.151
            ['h','h','h','m','l','l','l','m','l','m','m','m','h','m'] 
            # salary 0.029
        ])
        X = X.T
        Y = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
        result = get_best_split_point(X, Y, method='id3')
        self.assertEqual(result, 0)
        result = get_best_split_point(X, Y, method='id3',used_feature_list=[0])
        self.assertEqual(result, 1)
    # def test_build_tree(self):
    #     X = np.array([
    #         ['y','y','m','o','o','o','m','y','y','o','y','m','m','o'],
    #         [0,0,0,0,1,1,1,0,1,1,1,0,1,0]
    #     ])
    #     X = X.T
    #     Y = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
    #     node = build_tree(X,Y)





if __name__ == '__main__':
    unittest.main()

