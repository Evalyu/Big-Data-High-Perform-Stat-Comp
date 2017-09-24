import numpy as np
from random import shuffle
import sys
from numpy.linalg import inv

cpusmall = open("cpusmall_scaled.txt")
cpusmall = (cpusmall.readlines())

shuffle(cpusmall)
split = len(cpusmall) // 5
testing = cpusmall[ : split] # 20%
training = cpusmall[split : ] # 80%

yx = None

for i, line in enumerate(training):
    y_i = int(line.split()[0]) # 1 x 1
    x_i = list((y_i * float(x.split(":")[1]) for x in line.split()[1:])) # 1 x 12
    if i == 0:
        yx = np.matrix(x_i) # 1 x 12
    else:
        yx = np.add(yx, x_i)  # 1 x 12

Lambda = float(sys.argv[1])
identity = np.identity(12) # 12 x 12

xx = None

for i, line in enumerate(training):
    x_i = np.matrix(list((float(x.split(":")[1]) for x in line.split()[1:])))
    x_i_trans = np.transpose(x_i) # 12 x 1
    x = np.dot(x_i_trans, x_i) # 12 x 12
    if i == 0:
        xx = np.matrix(x) # 12 x 12
    else:
        xx = np.add(xx,x) # 12 x 12

xx_lamb_I = np.add(xx,identity*Lambda) # 12 x 12
xx_lamb_I = inv(xx_lamb_I) # 12 x 12

w_star = yx.dot(xx_lamb_I) # 1 x 12
# w_star = np.transpose(w_star) # 12 x 1

# w*, where lambda = 1, is
# [[ -0.28991301]
#  [  2.0416196 ]
#  [ -4.67194423]
#  [  6.17858662]
#  [ 20.19907527]
#  [-25.52965455]
#  [  7.39047632]
#  [ -6.12745516]
#  [ -7.89191553]
#  [-74.16419085]
#  [ -4.7811729 ]
#  [ 25.17649327]]

# print("w*, where lambda = 1, is", w_star)

MSE = None

for i, line in enumerate(testing):
    y_i = int(line.split()[0]) # 1 x 1
    x_i = np.matrix(list((float(x.split(":")[1]) for x in line.split()[1:]))) # 1 x 12
    x_w = x_i.dot(np.transpose(w_star)) # w_star: 12 x 1, x_i: 1 x 12 result: 1 x 1
    x_w_y = x_w - y_i # 1 x 1
    xwy2 = x_w_y * x_w_y # 1 x 1
    if i == 0:
        MSE = np.matrix(xwy2)
    else:
        MSE = np.add(MSE,xwy2) # 1 x 1

MSE = MSE/len(testing)
print("The MSE for lambda =", Lambda, "is", MSE)

# lambda = 0.01 [[94.6422998]]
# lambda = 0.1  [[97.34659095]]
# lambda = 1    [[104.59340661]]
# lambda = 10   [[101.54590039]]
# lambda = 100  [[112.24425021]]
