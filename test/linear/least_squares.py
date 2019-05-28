from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

from linear.least_squares import OwnLinearRegression

RANDOM_STATE = 5

b = datasets.load_breast_cancer()
(X_all, y_all) = b.data, b.target
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=RANDOM_STATE)

lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)

print("SCIKIT train score", lr.score(X_train, y_train))
print("SCIKIT test score", lr.score(X_test, y_test))

olr = OwnLinearRegression()
olr.fit(X_train, y_train)

print("OWN train score", olr.score(X_train, y_train))
print("OWN test score", olr.score(X_test, y_test))
