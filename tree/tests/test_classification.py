import unittest
import numpy as np

from pyml.tree.classification import entropy
from pyml.tree.classification import condition_entropy
from pyml.tree.classification import information_gain
from pyml.tree.classification import get_best_split_point
from pyml.tree.classification import divide_dataset
from pyml.tree.classification import DecisionTreeClassifier

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
    def test_divide_dataset(self):
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
        feature_values, Xs, Ys = divide_dataset(X,Y,0)

        test_Xs = [
            np.array([['m', '0', 'h'],
                                ['m', '1', 'l'],
                                ['m', '0', 'm'],
                                ['m', '1', 'h']]),
            np.array([['o', '0', 'm'],
                                ['o', '1', 'l'],
                                ['o', '1', 'l'],
                                ['o', '1', 'm'],
                                ['o', '0', 'm']]),
            np.array([['y', '0', 'h'],
                                ['y', '0', 'h'],
                                ['y', '0', 'm'],
                                ['y', '1', 'l'],
                                ['y', '1', 'm']])
        ]
        test_Ys = [
            np.array([1, 1, 1, 1]), 
            np.array([1, 1, 0, 1, 0]), 
            np.array([0, 0, 0, 1, 1])
        ]

        self.assertListEqual(feature_values, ['m','o','y'])
        for test_x, x in zip(test_Xs, Xs):
            np.testing.assert_array_equal(test_x, x)
        for test_y, y in zip (test_Ys, Ys):
            np.testing.assert_array_equal(test_y, y)

    def test_DecisionTreeClassifier_id3(self):
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
        dt_clf = DecisionTreeClassifier()
        dt_clf.fit(X,Y)
        y_pred = dt_clf.predict(X)
        test_y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1])
        np.testing.assert_array_equal(y_pred, test_y_pred)


if __name__ == '__main__':
    unittest.main()

