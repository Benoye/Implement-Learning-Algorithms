from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import OLS

from linear.successive_orthogonalization import OwnSuccessiveOrthogonalization

RANDOM_STATE = 5

b = datasets.load_breast_cancer()
(X_all, y_all) = b.data, b.target
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=RANDOM_STATE)

lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)

print("SCIKIT train score", lr.score(X_train, y_train))
print("SCIKIT test score", lr.score(X_test, y_test))

oso = OwnSuccessiveOrthogonalization()
oso.fit(X_train, y_train)

print("OWN train score", oso.score(X_train, y_train))
print("OWN test score", oso.score(X_test, y_test))

ols = OLS(y_train, X_train)
res = ols.fit()

print("STATSMODEL summary", res.summary())
