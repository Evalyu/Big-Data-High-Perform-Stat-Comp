import multiprocessing as mp
from sklearn.datasets import load_svmlight_file
import numpy as np
import time
import sys

def subProcess(Xtrain, ytrain, Xtest, ytest, shared_queue, min, max):
    """
    
    :type shared_queue: mp.Queue
    """
    acc = go_nn(Xtrain, ytrain, Xtest, ytest, min, max)
    shared_queue.put(acc)

def go_nn(Xtrain, ytrain, Xtest, ytest, min = 0, max = None):
    if max == None:
        max = Xtest.shape[0]
    correct = 0
    for i in range(min, max):  ## For all testing instances
        nowXtest = Xtest[i, :]
        ### Find the index of nearest neighbor in training data
        dis_smallest = np.linalg.norm(Xtrain[0, :] - nowXtest)
        idx = 0
        for j in range(1, Xtrain.shape[0]):
            dis = np.linalg.norm(nowXtest - Xtrain[j, :])
            if dis < dis_smallest:
                dis_smallest = dis
                idx = j
        ### Now idx is the index for the nearest neighbor

        ## check whether the predicted label matches the true label
        if ytest[i] == ytrain[idx]:
            correct += 1
    acc = correct / float(Xtest.shape[0])
    return acc

if __name__ == '__main__':
    traindata = load_svmlight_file("a9a.subset.train")
    testdata = load_svmlight_file("a9a.subset.test")
    Xtrain = traindata[0].todense()
    ytrain = traindata[1]
    Xtest = testdata[0].todense()
    ytest = testdata[1]

    if len(sys.argv) > 1:
        arg = int(sys.argv[1])
        Xtrain = Xtrain[0:arg]
        ytrain = ytrain[0:arg]

    start_time = time.time()

    sharedQueue = mp.Queue()
    coreCount = 4
    delta = Xtest.shape[0] // coreCount
    processes = list((mp.Process(target=subProcess,
                                 args=(Xtrain, ytrain, Xtest, ytest, sharedQueue, i * delta, (i + 1) * delta))
                      for i in range(coreCount - 1)))
    i = coreCount - 1
    processes += [mp.Process(target=subProcess,
                             args=(Xtrain, ytrain, Xtest, ytest, sharedQueue, i * delta, Xtest.shape[0]))]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    accuracy = 0

    while sharedQueue.empty() == False:
        single = sharedQueue.get()
        accuracy += single

    print("Multicore Accuracy %lf Time %lf secs.\n" % (accuracy, time.time() - start_time))

    start_time = time.time()
    acc = go_nn(Xtrain, ytrain, Xtest, ytest)

    print("Single Core Accuracy %lf Time %lf secs.\n" % (acc, time.time() - start_time))
    # print("Difference in accuracy: %lf " % (acc - accuracy))
