from sklearn import linear_model
from random import shuffle
from sklearn.metrics import mean_squared_error
import numpy as np

cpusmall = open("cpusmall_scaled.txt")
cpusmall = (cpusmall.readlines())

shuffle(cpusmall)
split = len(cpusmall) // 5
testing = cpusmall[ : split] # 20%
training = cpusmall[split : ] # 80%

values_train = []
targets_train = [] # 6554

for row in training:
    row = row.split()
    targets_train.append([int(row[0])])
    value_row_train = []
    for idx in row[1:]:
        value_train = float(idx.split(":")[1])
        value_row_train.append(value_train)
    values_train.append(value_row_train)

values_train = np.matrix(values_train) # 6554 x 12
targets_train = np.matrix(targets_train) # 6554 x 1

reg = linear_model.Ridge(max_iter = 50, alpha = 1)
y_pred = reg.fit(values_train, targets_train)
y_pred = y_pred.coef_

values_test = []
targets_test = []

for row in testing:
    row = row.split()
    targets_test.append([int(row[0])])
    value_row_test = []
    for idx in row[1:]:
        value_test = float(idx.split(":")[1])
        value_row_test.append(value_test)
    values_test.append(value_row_test)

values_test = np.matrix(values_test) # 1638 x 12
targets_test = np.matrix(targets_test) # 1638 x 1

y_true = reg.fit(values_train, targets_train)
y_true = y_true.coef_

print("pred:", y_pred)
print("true:", y_true)

print("MSE:",mean_squared_error(y_true, y_pred))
