import h5py
import numpy as np
def load_dataset():
    train_dataset = h5py.File('D:\Python\code\Deep-learning/5.cnn\cnn\datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('D:\Python\code\Deep-learning/5.cnn\cnn\datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

import math
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]               
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    
    num_complete_minibatches = math.ceil(m/mini_batch_size) 
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1) * mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k + 1) * mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    
    return mini_batches


def convert_to_one_hot(Y, C):
    
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


