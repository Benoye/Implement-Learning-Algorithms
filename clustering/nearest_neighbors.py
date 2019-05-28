import numpy as np


class OwnNearestNeighbors(object):
    def __init__(self, n_neighbors=5):
        self.X = None
        self.y = None
        self.number_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y

    def score(self, X, y):
        """
        zero-one
        """
        predictions = np.apply_along_axis(self.predict, 1, X)
        return 1.0 - np.absolute(predictions - y).sum() / y.size

    def predict(self, x):
        dist = np.apply_along_axis(self._euclidean_distance, 1, self.X, x)
        sorted_dist_indexes = np.argsort(dist)
        neighbors = sorted_dist_indexes[0:self.number_neighbors]
        mean_label = self.y[neighbors].mean()
        return self._group_prediction(mean_label)

    @staticmethod
    def _euclidean_distance(xi, xj):
        return np.linalg.norm(xi - xj)

    @staticmethod
    def _group_prediction(y_hat):
        if y_hat > 0.5:
            return 1
        else:
            return 0
