from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from clustering.nearest_neighbors import OwnNearestNeighbors

RANDOM_STATE = 5

b = datasets.load_breast_cancer()
(X_all, y_all) = b.data, b.target
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=RANDOM_STATE)

n = KNeighborsClassifier(n_neighbors=10)
n.fit(X_train, y_train)
print("SCIKIT train score", n.score(X_train, y_train))
print("SCIKIT test score", n.score(X_test, y_test))

onn = OwnNearestNeighbors(n_neighbors=10)
onn.fit(X_train, y_train)
print("OWN train score", onn.score(X_train, y_train))
print("OWN train score", onn.score(X_test, y_test))
