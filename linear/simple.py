import numpy as np


class OwnSimpleLinearRegression(object):
    def __init__(self):
        self.alpha = None
        self.beta = None

    def fit(self, X, y):
        tmp = np.column_stack((X, y)).T
        cov = np.cov(tmp, bias=True)  # option here is to normalize by sample length
        var = np.var(X)
        self.beta = cov[0][1] / var
        self.alpha = y.mean() - self.beta * X.mean()

    def score(self, X, y):
        """
        R^2
        """
        y_hat = np.apply_along_axis(self.predict, 0, X).T
        u = np.sum((y - y_hat) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v

    def predict(self, x):
        return self.beta * x + self.alpha
