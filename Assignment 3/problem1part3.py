import numpy as np
import random
import sys
import warnings

warnings.filterwarnings('error')

# Compute the gradient
# Time complexity: O(nnz(X)) per iteration

max_col = 12
w_0 = np.random.randn(max_col) # 1 x 12
w_0 = np.matrix(w_0).transpose() # 12 x 1
step_Size = sys.argv[1]

cpusmall = open("cpusmall_scaled.txt")
cpusmall = (cpusmall.readlines())

random.shuffle(cpusmall)
split = len(cpusmall) // 5
testing = cpusmall[: split]  # 20%
training = cpusmall[split:]  # 80%

def gradient_fix(data_training, step_size, stop_cond, init_soln):

    w = init_soln # 12 x 1
    Lambda = 1

    y = None

    for i, line in enumerate(data_training):
        y_i = int(line.split()[0])  # 1 x 1
        if i == 0:
            y = np.matrix(y_i)  # n x 1
        else:
            y = np.vstack([y_i, y])  # n x 1

    X = None

    for i, line in enumerate(data_training):
        x_i = np.matrix(list((float(x.split(":")[1]) for x in line.split()[1:])))
        if i == 0:
            X = np.matrix(x_i)  # 1 x 12
        else:
            X = np.vstack([X, x_i])  # 1 x 12

    # 12 x 1
    grad_f_of_w = np.transpose(X).dot(X.dot(w) - y) + (Lambda * w) # type: np.matrixlib.defmatrix.matrix

    r_0 = np.sqrt(np.transpose(grad_f_of_w).dot(grad_f_of_w)) # 1 x 1

    for i in range(50):
        # 12 x 1
        g: np.matrixlib.defmatrix.matrix = np.transpose(X).dot(X.dot(w) - y) + (Lambda * w) # type: np.matrixlib.defmatrix.matrix
        # 1 x 1 <= 1 x 1
        if (np.sqrt(np.transpose(g).dot(g)) <= (stop_cond * r_0))[0,0] == True:
            break
        # g_step_calced = False
        g_step = float(step_size) * g
        w = np.subtract(w , g_step)

    return w

print(gradient_fix(training, step_Size, 0.001, w_0))

w_star = gradient_fix(training, step_Size, 0.001, w_0) # 12 x 1

MSE = None

for i, line in enumerate(testing):
    y_i = int(line.split()[0]) # 1 x 1
    x_i = np.matrix(list((float(x.split(":")[1]) for x in line.split()[1:]))) # 1 x 12
    x_w = x_i.dot(w_star) # w_star: 12 x 1, x_1: 1 x 12 result: 1 x 1
    x_w_y = x_w - y_i # 1 x 1
    xwy2 = x_w_y * x_w_y # 1 x 1
    if i == 0:
        MSE = np.matrix(xwy2)
    else:
        MSE = np.add(MSE,xwy2) # 1 x 1

MSE = MSE/len(testing)

print("The MSE for step size =", step_Size, "is", MSE)

# step_size = 0.0000001
# [[-0.33029413]
#  [-1.35700764]
#  [-1.13281849]
#  [-1.10394472]
#  [-3.32495793]
#  [-1.0979192 ]
#  [-1.83684214]
#  [-2.60568559]
#  [-2.89032065]
#  [-3.86823107]
#  [-3.44392521]
#  [ 0.13587654]]
# The MSE for step size = 0.0000001 is [[ 4411.71166547]]
#
# step_size = 0.000001
# [[-10.20202987]
#  [ -6.40153182]
#  [ -4.67756108]
#  [ -9.64074521]
#  [ -7.09951138]
#  [ -8.60610341]
#  [ -9.20698297]
#  [ -7.48503728]
#  [ -8.55147932]
#  [ -9.33911636]
#  [ -7.23760331]
#  [  1.92411671]]
# The MSE for step size = 0.000001 is [[ 275.54406627]]
#
# step_size = 0.00001
# [[-10.1970878 ]
#  [-11.01040163]
#  [ -4.65171718]
#  [-11.19945415]
#  [-10.69438669]
#  [ -5.06482119]
#  [ -7.81266456]
#  [ -6.28470961]
#  [ -7.02247807]
#  [-12.27640714]
#  [ -6.58399817]
#  [ -0.34176006]]
# The MSE for step size = 0.00001 is [[ 288.71192157]]
#
# step_size = 0.0001
# [[  1.94326758e+34]
#  [  1.89677388e+34]
#  [  1.30831610e+34]
#  [  1.83817509e+34]
#  [  1.88392669e+34]
#  [  1.63729234e+34]
#  [  1.80979923e+34]
#  [  1.69193601e+34]
#  [  1.78627580e+34]
#  [  1.95967689e+34]
#  [  1.41606998e+34]
#  [ -3.84737746e+33]]
# The MSE for step size = 0.0001 is [[  2.93133069e+70]]
#
# step_size = 0.001
# [[  1.32457025e+88]
#  [  1.29247613e+88]
#  [  8.88182528e+87]
#  [  1.25130655e+88]
#  [  1.28334667e+88]
#  [  1.11605476e+88]
#  [  1.23405360e+88]
#  [  1.15191001e+88]
#  [  1.21667251e+88]
#  [  1.33592241e+88]
#  [  9.68189955e+87]
#  [ -2.56971468e+87]]
# The MSE for step size = 0.001 is [[  1.36577868e+178]]
#
# step_size = 0.01
# [[  2.83832074e+138]
#  [  2.76888031e+138]
#  [  1.90226426e+138]
#  [  2.68472346e+138]
#  [  2.75216603e+138]
#  [  2.38817672e+138]
#  [  2.64351532e+138]
#  [  2.47164326e+138]
#  [  2.60965633e+138]
#  [  2.86285394e+138]
#  [  2.08248842e+138]
#  [ -5.45140375e+137]]
# The MSE for step size = 0.01 is [[  6.26806087e+278]]