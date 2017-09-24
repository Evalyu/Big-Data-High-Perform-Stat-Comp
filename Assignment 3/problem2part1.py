import numpy as np

# def f(w):
#     # w is
#     f_w = 0.5 * Lambda * w^T * w
#     for row in DATA:
#         y = row[0]
#         x = row[1:]
#         antilog = 1 + exp(-y * x * w) # antilog is antilogarithm, which is the name of the input in logarithm
#         f_w = f_w + log(antilog)
#     return f_w

# function f, which takes w as input and output f of w
def f_of_w(X, y, Lambda, init_sol):
    w = init_sol # 1355191 x 1
    f_w = (0.5 * Lambda) * (np.transpose(w) * w)  # 1 x 1
    for i in range(X.shape[0]):
        y_i = float(y[i]) # 1 x 1
        x_i = X[i] # 1 x 1355191
        # antilog is antilogarithm, which is the name of the input in logarithm
        antilog = 1 + np.exp(-y_i * x_i * w) # 1 x 1
        f_w = np.add(f_w, antilog) # 1 x 1
    return f_w

# def grad_f(w):
#     # w is n x 1
#     grad_f_w = Lambda * w^T # 1 x n
#     for row in DATA:
#         y = row[0] # 1 x 1
#         x = row[1:] # 1 x n
#         tmp = y * x # tmp is temporary, which is used to hold some terms for slight efficiency in the code, 1 x n
#         den = 1 + exp(-y * x * w^T) # den is denominator
#         grad_f_w = grad_f_w + (1/den - 1) * tmp
#     return grad_f_w # 1 x n or n x 1
def gradient_f_of_w(X, y, int_sol, Lambda):
    w = int_sol # 62061 x 1

    grad_f_of_w = Lambda * np.transpose(w) # 1 x 62061

    for i in range(X.shape[0]):
        y_i = float(y[i])  # 1 x 1
        x_i = X[i]  # 1 x 62061
        # temp is temporary, which is used to hold some terms for slight efficiency in the code, 1 x 1355191
        temp = y_i * x_i # 1 x 1355191
        # den is denominator
        denom = 1 + np.exp(-y_i * x_i * w) # y_i: 1 x 1, x_i: 1 x 1355191, w: 1355191 x 1 => result: 1 x 1
        grad_f_of_w = np.add(grad_f_of_w,((1/denom - 1) * temp)) # 1 x 1355191

    return grad_f_of_w # 1 x 62061
