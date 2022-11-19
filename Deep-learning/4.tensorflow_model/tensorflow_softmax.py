import tensorflow as tf

#创建占位符
def create_placeholder(n_x, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None])
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None])

    return X, Y

#参数初始化
def initial_params(layer_size):
    L = len(layer_size)
    params = {}

    for l in range(1, L):
        params['W'+ str(l)] = tf.get_variable(name='W'+ str(l), shape=[layer_size[l], layer_size[l-1]], initializer=tf.initializers.glorot_uniform(seed=1), regularizer=tf.keras.regularizers.l2(0.03))
        params['b'+ str(l)] = tf.get_variable(name='b'+ str(l), shape=[layer_size[l], 1], initializer=tf.zeros_initializer())
    
    return params

#前传到线性输出
def forward_propagation(X, params):
    L = len(params) // 2
    A_prev = X

    for l in range(1, L):
        Z = tf.add(tf.matmul(params['W'+ str(l)], A_prev), params['b'+ str(l)])
        A_prev = tf.nn.relu(Z)
    
    Z = tf.add(tf.matmul(params['W'+ str(L)], A_prev), params['b'+ str(L)])

    return Z

#计算代价
def compute_cost(Z, Y):
    Z = tf.transpose(Z)
    Y = tf.transpose(Y)
    cross_entropy_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z))
    l2_reg_cost = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cost = cross_entropy_cost  + l2_reg_cost
    
    
    return cost

import h5py
import numpy as np
def load_dataset():
    train_dataset = h5py.File('D:\Python\code\Deep-learning/4.tensorflow_model\datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) 

    test_dataset = h5py.File('D:\Python\code\Deep-learning/4.tensorflow_model\datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:]) 
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = load_dataset()

#展开数据成(features, m)
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

#归一化
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

#随机小批量
import math
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[1]                 
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    num_minibatches = math.ceil(m/mini_batch_size) 
    for k in range(num_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

#独热编码
C = len(classes)
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
Y_train = convert_to_one_hot(Y_train_orig, C)
Y_test = convert_to_one_hot(Y_test_orig, C)

'''组建tensorflow model'''
import matplotlib.pyplot as plt
def tensorflow_model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, iterations=1500, minibatch_size=32, print_cost=True, plot=True):
    tf.reset_default_graph()
    tf.set_random_seed(1)

    n_x, n_y, m = X_train.shape[0], Y_train.shape[0], Y_train.shape[1]
    layer_size = [n_x, 25, 12, n_y]
    

    X, Y = create_placeholder(n_x, n_y)
    params = initial_params(layer_size)
    Z = forward_propagation(X, params)
    cost = compute_cost(Z, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    costs = []
    seed = 0
    num_minibatch = math.ceil(m / minibatch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #pre_iter_cost = float('inf')
        for i in range(iterations):
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            cur_iter_cost = 0
            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                cur_iter_cost = cur_iter_cost + minibatch_cost / num_minibatch

            if i % 10 == 0:
                costs.append(cur_iter_cost)
                if print_cost and i % 100 == 0:
                    print('第',i,'次迭代代价为:',cur_iter_cost)

                    #if cur_iter_cost > pre_iter_cost:
                    #   break   
                    #pre_iter_cost = cur_iter_cost

        if plot:
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('iterations(per 10)')
            plt.show()

        params = sess.run(params)

        True_predictions = tf.equal(tf.argmax(Z, axis=0), tf.argmax(Y, axis=0))
        accuracy = tf.reduce_mean(tf.cast(True_predictions, tf.float32))

        print('训练集精度:',sess.run(accuracy, feed_dict={X: X_train, Y: Y_train}))
        print('测试集精度:',sess.run(accuracy, feed_dict={X: X_test, Y: Y_test}))

        return params

params = tensorflow_model(X_train, Y_train, X_test, Y_test)
