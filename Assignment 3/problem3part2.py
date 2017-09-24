from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

mem = Memory("./mycache4")

@mem.cache
def get_data(data):
    my_data = load_svmlight_file(data)
    return my_data[0], my_data[1]

X_train, y_train = get_data("E2006.train.bz2") # X_train: 16087 x 150360
X_test, y_test = get_data("E2006.test.bz2")

y_test = np.matrix(y_test)
y_test = np.transpose(y_test)


X_train = X_train[:, :-2] # 16087 x 150358
y_train = np.matrix(y_train) # 16087 x 1
y_train = np.transpose(y_train)

reg = linear_model.Ridge(max_iter = 50, alpha = 1)
y_pred = reg.fit(X_train, y_train)
y_pred = y_pred.coef_

y_true = reg.fit(X_test, y_test)
y_true = y_true.coef_

print("pred:", y_pred)
print("true:", y_true)

print("MSE:",mean_squared_error(y_true, y_pred))

