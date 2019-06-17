import numpy as np
from sklearn import linear_model

from linear.simple import OwnSimpleLinearRegression

X = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]).reshape(-1 ,1)
y = np.array([245, 312, 279, 308, 199, 219, 405, 324, 319, 255])

lr = linear_model.LinearRegression()
lr.fit(X, y)

print("SCIKIT score", lr.score(X, y))

oslr = OwnSimpleLinearRegression()
oslr.fit(X, y)

print("OWN score", oslr.score(X, y))
