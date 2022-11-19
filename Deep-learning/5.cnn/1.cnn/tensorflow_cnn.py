import numpy as np

'''边缘填充'''
def  padding(X, pad):

    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)))

    return X_pad

'''单次卷积'''
def conv_single_step(a_prev_slice, W, b):
    s = np.multiply(a_prev_slice, W) + b
    Z = np.sum(s)

    return Z

'''卷积'''
def conv_forward(A_prev, W, b, hparams):
    stride, pad =  hparams['stride'], hparams['pad']

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape

    f, f, n_C_prev, n_C = W.shape

    n_H = int((n_H_prev + 2 * pad - f) / stride + 1)
    n_W = int((n_W_prev + 2 * pad - f) / stride + 1)

    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = padding(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h * stride
                    v_end = h * stride + f
                    h_start = w * stride
                    h_end = w * stride + f
                    a_prev_slice = a_prev_pad[v_start:v_end, h_start:h_end,:]
                    Z[i,h,w,c] = conv_single_step(a_prev_slice, W[:,:,:,c], b[0,0,0,c])
    cache = (A_prev, W, b, hparams)

    return Z, cache

'''池化'''
def pooling_forward(A_prev, hparams, mode='max'):
    m, n_H_prev, n_W_prev, n_C = A_prev.shape

    f, stride = hparams['f'], hparams['stride']

    n_H = int((n_H_prev - f) / stride + 1)
    n_W = int((n_W_prev - f) / stride + 1)

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h * stride
                    v_end = h * stride + f
                    h_start = w * stride
                    h_end = w * stride + f
                    a_prev_slice = a_prev[v_start:v_end, h_start:h_end, c]

                    if mode == 'max':
                        A[i,h,w,c] = np.max(a_prev_slice)
                    else:
                        A[i,h,w,c] = np.mean(a_prev_slice)
    
    cache = (A_prev, hparams)

    return A, cache

'''反向传播--卷积层'''
def conv_backward(dZ, cache):
    A_prev, W, b, hparams = cache
    m, n_H, n_W, n_C = dZ.shape

    pad , stride = hparams['pad'], hparams['stride']

    f, f, n_C_prev, n_C = W.shape

    A_prev_pad = padding(A_prev, pad)
    dA_prev_pad = np.zeros((A_prev_pad.shape))
    dW = np.zeros((W.shape))
    db = np.zeros((b.shape))

    for i in range(m):
        for c in range(n_C):
            for h in range(n_H):
                for w in range(n_W):
                    v_start = h * stride
                    v_end = h * stride + f
                    h_start = w * stride
                    h_end = w * stride + f

                    a_prev_slice =  A_prev_pad[i, v_start:v_end, h_start:h_end, :]
                    filter = W[:,:,:,c]

                    dW[:,:,:,c] += dZ[i, h, w, c] * a_prev_slice
                    db[:,:,:,c] += dZ[i, h, w, c]

                    dA_prev_pad[i, v_start:v_end, h_start:h_end, :] += dZ[i, h, w, c] * filter
    
    dA_prev = dA_prev_pad[:,pad:-pad, pad:-pad,:]

    return dA_prev, dW, db

'''反向传播--池化层'''
#最大池化：记录最大值位置
def record_location(X):
    loc = (X == np.max(X))

    return loc

#均值池化：分流梯度
def distribute_grad(dz, shape):
    H, W = shape
    avg = np.ones(shape) * dz / (H * W)

    return avg

def pooling_backward(dA, cache, mode='max'):
    A_prev, hparams = cache
    f, stride = hparams['f'], hparams['stride']

    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h * stride
                    v_end = h * stride + f
                    h_start = w * stride
                    h_end = w * stride + f

                    a_prev_slice = A_prev[i, v_start:v_end, h_start:h_end, c]
                    if mode == 'max':
                        loc = record_location(a_prev_slice)
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += dA[i,h,w,c] * loc
                    else:
                        avg = distribute_grad(dA[i,h,w,c], (f,f))
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += avg
    return dA_prev

'''Tensorflow model'''
import tensorflow as tf

#创建占位符
def create_placeholder(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, (None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, (None, n_y))

    return X, Y

#初始化参数变量
def initialize_params():
    tf.set_random_seed(1)
    
    W1 = tf.get_variable("W1",[4,4,3,8],initializer=tf.glorot_uniform_initializer(seed=0))
    W2 = tf.get_variable("W2",[2,2,8,16],initializer=tf.glorot_uniform_initializer(seed=0))
    
    params = {"W1": W1,
              "W2": W2}
    return params

#前向传播
def forward_propagation(X, params):
    W1 = params['W1']
    W2 = params['W2']

    Z1 = tf.nn.conv2d(X, filter=W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')

    Z2 = tf.nn.conv2d(P1, filter=W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

    P = tf.keras.layers.Flatten()(P2)
    Z = tf.keras.layers.Dense(6)(P)

    return Z 

#计算成本
def compute_cost(Z, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z))

    return cost


#建立模型
import math
import cnn_utils
import matplotlib.pyplot as plt
def tensorflow_cnn_model(X_train, Y_train, X_test, Y_test, num_epochs=100, learning_rate=0.008, minibatch_size=64, print_cost=False, plot=True):
    tf.reset_default_graph()

    m, n_H0, n_W0, n_C0 = X_train.shape
    n_y = Y_train.shape[1]

    X, Y = create_placeholder(n_H0, n_W0, n_C0, n_y)
    params = initialize_params()
    Z = forward_propagation(X, params)
    cost = compute_cost(Z, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        costs = []
        num_minibatch = math.ceil(m / minibatch_size)
        seed = 0
        for epoch in range(num_epochs):
            seed = seed + 1
            minibatches = cnn_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)
            epoch_cost = 0
            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatch
            
            if epoch % 5  == 0:
                costs.append(epoch_cost)
                if print_cost and epoch % 10 == 0:
                    print('第',epoch,'次 迭代,代价为:',epoch_cost)
        if plot:
            plt.plot(costs)
            plt.title('learning rate = '+ str(learning_rate))
            plt.xlabel('iterations (per 5)')
            plt.ylabel('cost')
            plt.show()
        
        correct_prediction = tf.equal(tf.argmax(Z, axis=1), tf.argmax(Y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print('训练集准确率:', accuracy.eval({X: X_train, Y: Y_train}))
        print('测试集准确率:', accuracy.eval({X: X_test, Y: Y_test}))

        return params


train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = cnn_utils.load_dataset()
X_train = train_set_x_orig / 255
X_test = test_set_x_orig / 255

C = len(classes)
Y_train = cnn_utils.convert_to_one_hot(train_set_y_orig, C)
Y_test = cnn_utils.convert_to_one_hot(test_set_y_orig, C)

parameters = tensorflow_cnn_model(X_train, Y_train, X_test, Y_test, learning_rate=0.005, num_epochs=200, print_cost= True)
 