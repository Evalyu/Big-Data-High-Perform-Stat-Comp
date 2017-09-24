from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

mem = Memory("./mycache5")

@mem.cache
def get_data(data):
    my_data = load_svmlight_file(data)
    return my_data[0], my_data[1]

X, y = get_data("news20.binary.bz2")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
y_train =  np.matrix(y_train).transpose() # 15996 x 1
y_test = np.matrix(y_test).transpose() # 4000 x 1

ols = linear_model.LogisticRegression(max_iter = 50)
y_pred = ols.fit(X_train, y_train)
y_pred = y_pred.coef_ # 1 x 1355191

y_true = ols.fit(X_test, y_test)
y_true = y_true.coef_ # 1 x 1355191

print("MSE:",mean_squared_error(y_true, y_pred))