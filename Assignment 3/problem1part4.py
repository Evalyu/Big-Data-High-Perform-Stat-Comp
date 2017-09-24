import numpy as np
import scipy.sparse as sparse
import sys
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

# Remove last two columns of train to match the # of columns in test
# E2006.train is 16087 x 150360 ==> 16087 x 150358
# E2006.test is 3308 x 150358

mem = Memory("./mycache")

@mem.cache
def get_data(data):
    my_data = load_svmlight_file(data)
    return my_data[0], my_data[1]

X_train, y_train = get_data("E2006.train.bz2")
X_test, y_test = get_data("E2006.test.bz2")

# Remove last two columns of X_train so it has the same number of columns as X_test
X_train = sparse.csr_matrix(X_train[:, :-2])

step_Size = sys.argv[1]
eta = 0.001
max_col = 150358
w_0 = np.random.randn(max_col) # 1 x 150358
w_0 = np.matrix(w_0).transpose() # 150358 x 1

# Algorithm 1 Gradient Descent with Fixed Step Size
def gradient_fix(X, y, step_size, stop_cond, init_soln):
    w = init_soln # 150358 x 1
    Lambda = 1.0
    y =  np.matrix(y).transpose() # 16087 x 1

    # X: 16087 x 150358
    # y: 16087 x 1
    # w: 150358 x 1
    # grad_f_of_w: 150358 x 1
    grad_f_of_w = np.transpose(X).dot(X.dot(w) - y) + (Lambda * w)  # type: np.matrixlib.defmatrix.matrix

    r_0 = np.sqrt(np.transpose(grad_f_of_w).dot(grad_f_of_w)) # 1 x 1

    for i in range(50):
        g : np.matrixlib.defmatrix.matrix  = np.transpose(X).dot(X.dot(w) - y) + (Lambda * w) # 150358 x 1
        if (np.sqrt(np.transpose(g).dot(g)) <= (stop_cond * r_0))[0,0] == True: # 1 x 1 <= 1 x 1
            break
        w = np.subtract(w , float(step_size) * g)

    return w

w_star = gradient_fix(X_train, y_train, step_Size, eta, w_0) # 1 x 150358

MSE = None

for i in range(X_test.shape[0]):
    y_i = float(y_test[i])
    x_i = X_test[i]
    x_w = x_i.dot(w_star) # w_star: 3 x 1
    x_w_y = x_w - y_i # 1 x 1
    xwy2 = x_w_y * x_w_y # 1 x 1
    if i == 0:
        MSE = np.matrix(xwy2)
    else:
        MSE = np.add(MSE,xwy2) # 1 x 1

MSE = MSE/X_test.shape[0]

print("The MSE for step size =", step_Size, "is", MSE)

# The MSE for step size = 0.0000001 is [[ 9.45616772]]
# The MSE for step size = 0.000001 is [[ 0.15474644]]
# The MSE for step size = 0.00001 is [[ 0.15156958]]
# The MSE for step size = 0.0001 is [[  3.97646316e+124]]
# The MSE for step size = 0.001 is [[  3.39087729e+227]]
# The MSE for step size = 0.01 is [[ inf]]