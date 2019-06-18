import numpy as np
from sklearn.preprocessing import StandardScaler

from linear.gram_schmidt_multiple_passes import OwnSuccessiveOrthogonalization


class GramSchmidtProcedure(object):
    """
    Gram-Schmidt procedure for Multiple Regression
    Implementation for single pass
    """

    def __init__(self):
        self.z_terms = None
        self.scaler = None
        self.beta_terms = None
        self.gamma_matrix = None

    def fit(self, X, y):
        p = X.shape[1]
        self.gamma_matrix = np.zeros((p, p))
        x_0 = np.ones(X.shape[0])
        self.z_terms = [x_0]

        X_std = self._standardize(X)
        for j in range(1, p):
            x_j = X_std[:, j-1]

            z_j = x_j
            for k, z_k in enumerate(self.z_terms):
                self.gamma_matrix[k][j] = self._regress(x_j, z_k)[0]
                z_j -= self._regress(x_j, z_k)[0] * z_k
            self.z_terms.append(z_j)

        z_p = self.z_terms[-1]
        beta_p = self._regress(y, z_p)[0]

        self.beta_terms = np.empty((p,))
        self.beta_terms[p-1] = beta_p
        for j in range(p-1, 0, -1):
            z_jprev = self.z_terms[j-1]
            g = self.gamma_matrix[j-1][j]
            b = self.beta_terms[j]

            beta = self._compute_ip(z_jprev, y) / np.linalg.norm(z_jprev) ** 2 - g * b
            self.beta_terms[j-1] = beta

    def score(self, X, y):
        """
        R^2
        """
        y_hat = np.apply_along_axis(self._apply_model, 1, X)
        u = np.sum((y - y_hat) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v

    def predict(self, x):
        return np.dot(x, self.beta_terms)

    def _regress(self, a, b):
        """
        Regress a on b or "b is adjusted for a" or "b is orthogonalized with respect to a"
        """
        beta = self._compute_ip(b, a) / self._compute_ip(a, a)
        residual = b - np.dot(a, beta)
        return beta, residual

    def _apply_model(self, x):
        x_std = self.scaler.transform([x])[0]
        return self._group_prediction(self.predict(x_std))

    def _standardize(self, X):
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self.scaler.transform(X)

    @staticmethod
    def _compute_ip(a, b):
        """
        compute inner product
        """
        return np.dot(a.T, b)

    @staticmethod
    def _group_prediction(y_hat):
        if y_hat > 0.5:
            return 1
        return 0
