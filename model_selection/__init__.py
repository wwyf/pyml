import numpy as np
class ShuffleSplit():
    """
    Parameters
    -----------
    n_splits : the number of samples

    test_size : float
        represent the proportion of the dataset to include in the test split.

    """
    def __init__(self, n_splits=10, test_size=0.1, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = 1-test_size
        if (random_state is not None):
            np.random.seed(random_state)


    def split(self, X):
        """Generate indices to split data into training and test set

        Parameters
        ----------
        X : array-like, shape(n_samples, n_features)

        Yields
        ---------
        train : ndarray
            The training set indices for that split
        test : ndarray
            The testing set indices for that split

        """
        n_samples = X.shape[0]
        n_train = int(n_samples * self.train_size)
        n_test = int(n_samples * self.test_size)
        all_index = np.arange(0, n_train+n_test)
        for i in range(0, self.n_splits):
            np.random.shuffle(all_index)
            yield all_index[:n_train],all_index[n_train:]

    def get_n_splits(self):
        """
        Returns
        -----------
        n_splits : int

        """
        return self.n_splits

if __name__ == '__main__':
    X = np.array([1,2,3,4,5,6,7,8,9,90])
    rs = ShuffleSplit(test_size=0.3)
    for x_i, y_i in rs.split(X):
        print("train_index : {}, test_index : {}".format(x_i, y_i))