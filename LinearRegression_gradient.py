import numpy as np
from sklearn.preprocessing import StandardScaler

class LinearRegression_gradient():
    def __init__(self, learning_rate=0.001, iterations=700):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # normalizacja
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # gradient descent
        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            diff = y_predicted - y
            weight_gradient = (1 / n_samples) * np.dot(X.T, diff)
            bias_gradient = (1 / n_samples) * np.sum(diff)

            # parametry
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

    def predict(self, X):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return np.dot(X, self.weights) + self.bias