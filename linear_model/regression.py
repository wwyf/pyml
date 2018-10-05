import numpy as np

class LinearRegression():
    def  __init__(self, learning_rate=0.1, num_iterations=2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = {}
        self.information = {}

    def initialize_parameters(self, n_features):
        """
        initialize the parameters with zeros

        w -- initialized vector of shape (n_features+1, 1)

        """
        parameters = {}
        # TODO: 参数如何初始化
        # parameters['w'] = np.random.normal(size=(n_features, 1))
        # parameters['b'] = np.random.normal()
        parameters['w'] = np.zeros((n_features+1, 1))
        return parameters

    def propagate(self, w, X, Y):
        """
        Parameters
        ----------
        w : 2d array-like shape(n_features+1, 1)
        X : 2d array-like shape(n_samples, n_features+1)
        Y : 1d array-like shape(n_samples, 1)
        """
        m = X.shape[0] # m = n_samples
        n_features_plus = X.shape[1]
        Y_hat = (w.T * X).sum(axis=1).reshape(m,1) # Y_hat shape(n_samples, 1)
        cost = 1/(2*m) * np.square(Y_hat-Y).sum() # 
        dw = np.zeros((n_features_plus,1))
        for j in range(0, n_features_plus):
            # 计算n_features个参数对应的梯度
            dw[j,0] = 1/m * ((Y_hat-Y)*(X[:,j].reshape(-1,1))).sum()
        cost = np.squeeze(cost)
        grads = {"dw": dw}
        return grads, cost

    def optimize(self, w, X, Y, num_iterations, print_cost=False):
        """
        Parameters
        ----------
        w : 2d array-like shape(n_features+1, 1)
        X : 2d array-like shape(n_samples, n_features+1)
        Y : 1d array-like shape(n_samples, 1)
        """
        learning_rate = self.learning_rate
        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(w, X, Y)
            dw = grads["dw"]
            w = w - learning_rate * dw
            if i % 100 == 0:
                costs.append(cost)
            # Print the cost every 100 training examples
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        params = {"w": w}
        grads = {"dw": dw}
        return params, grads, costs

    def fit(self, X, Y, watch=False):
        """
        Parameters
        ------------
        X : 2d array-like (n_samples, n_features)
        Y : 1d array-like (n_samples, )
        
        """
        n_features = X.shape[1]
        n_samples = X.shape[0]
        parameters = self.initialize_parameters(n_features)
        w = parameters['w']
        X_reshaped = np.hstack([np.ones((n_samples, 1)), X]) 
        # shape(n_samples, n_features+1)
        Y_reshaped = Y.reshape((-1,1))
        # shape(n_samples, 1)
        parameters, grads, costs = self.optimize(w, X_reshaped, Y_reshaped, self.num_iterations, print_cost=watch)

        self.information['grads'] = grads
        self.information['costs'] = costs
        self.parameters = parameters
        return

    def predict(self, X_pred):
        """
        Parameters
        ------------
        X_pred : 2d array-like (n_samples, n_features)

        Retures
        ---------
        Y_pred : 1d array-like (n_samples, )
        """
        m = X_pred.shape[0]
        n_features = X_pred.shape[1]
        w = self.parameters['w'] # shape(n_features+1, 1)
        X = np.hstack([np.ones((m, 1)), X_pred])
        # shape(n_samples, n_features+1)
        Y_hat =  (w.T * X).sum(axis=1).reshape(-1)
        return Y_hat

if __name__ == '__main__':
    # 经验： 数据要先归一化再进行回归，不然很容易不收敛
    def f(x):
        w = np.array([
            [4,5]
        ])
        return np.dot(X,w.T) + 10 + np.random.normal()/1000
    X = np.hstack([30 * np.random.random_sample((100,1))-10,2 * np.random.random_sample((100,1))-2])

    # X = np.array([
    #     [1],
    #     [2]
    # ])
    Y = f(X)
    print(X.shape)
    print(Y.shape)
    lr = LinearRegression(learning_rate=0.01, num_iterations=10000)
    print(lr.parameters)
    lr.fit(X,Y, watch=True)
    print(lr.parameters)