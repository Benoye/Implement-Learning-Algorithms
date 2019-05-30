import numpy as np

from numpy.linalg import inv


class OwnLinearRegression(object):
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        X_t = X.T
        self.beta = np.dot(np.dot(inv(np.dot(X_t, X)), X_t), y)

    def score(self, X, y):
        """
        R^2
        """
        y_hat = np.apply_along_axis(self._apply_model, 1, X, self.beta)
        u = np.sum((y - y_hat) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v

    def predict(self, x):
        return np.dot(x.T, self.beta)

    def _apply_model(self, x, model):
        return self._group_prediction(np.dot(x.T, model))

    @staticmethod
    def _group_prediction(y_hat):
        if y_hat > 0.5:
            return 1
        return 0
