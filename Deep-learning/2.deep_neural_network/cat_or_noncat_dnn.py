from dnn_utils import sigmoid,sigmoid_backward, relu, relu_backward
import lr_utils

import numpy as np

#参数初始化
def deep_layer_initial(layers_size):
    np.random.seed(7)
    params = {}
    L = len(layers_size)
    
    for l in range(1,L):
        params["W" + str(l)] = np.random.randn(layers_size[l], layers_size[l - 1]) / np.sqrt(layers_size[l - 1])
        params["b" + str(l)] = np.zeros((layers_size[l], 1))
        
        
    return params

#前向传播
def forward_propagation(X, params):
    L = len(params) // 2
    caches = {}
    A_prev = X
    caches['A0'] = X
    for l in range(1, L + 1):
        W_l, b_l = params['W'+ str(l)], params['b'+ str(l)]
        Z_l = np.dot(W_l, A_prev) + b_l
        A_l, _ =  relu(Z_l) if l != L else sigmoid(Z_l)
        A_prev = A_l
        caches['W'+ str(l)], caches['Z'+ str(l)], caches['A'+ str(l)] = W_l, Z_l, A_l
    
    AL = A_l

    return AL, caches

#计算成本
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = 1/m * np.sum(-np.multiply(Y, np.log(AL)) - np.multiply(1-Y, np.log(1-AL)))

    return cost

#反向传播
def back_propagation(caches, Y):
    L = (len(caches) - 1) // 3
    grads = {}
    m = Y.shape[1]
    for l in range(L, 0, -1):
        A_l = caches['A'+ str(l)]
        Z_l = caches['Z'+ str(l)]
        W_l = caches['W'+ str(l)]
        if l == L:
            dA_l = -np.divide(Y, A_l) + np.divide((1-Y), (1-A_l))
            dZ_l = sigmoid_backward(dA_l, Z_l)
        else:
            dZ_l = relu_backward(dA_prev, Z_l)
        A_prev = caches['A' + str(l - 1)]
        dW_l = 1/m * np.dot(dZ_l, A_prev.T)
        db_l = 1/m * np.sum(dZ_l, axis=1, keepdims=True)
        dA_prev = np.dot(W_l.T, dZ_l)

        grads['dW'+ str(l)] , grads['db'+ str(l)] = dW_l , db_l
    
    return grads

#更新参数
def update_params(params, grads, learning_rate):
    L = len(params) // 2
    for l in range(1, L + 1):
        params['W'+ str(l)] -= learning_rate * grads['dW'+ str(l)]
        params['b'+ str(l)] -= learning_rate * grads['db'+ str(l)]
    
    return params

#组建模型
import matplotlib.pyplot as plt
def deep_nn(X, Y, layers_size, iterations=3000, learning_rate=0.0075, print_cost=False, plot=False):
    params = deep_layer_initial(layers_size)
    costs =[]

    for i in range(iterations):
        AL, caches = forward_propagation(X, params)
        grads = back_propagation(caches, Y)
        params = update_params(params, grads, learning_rate)

        if i % 100 == 0:
            cost = compute_cost(AL, Y)
            costs.append(cost)
            if print_cost and i % 1000 == 0:
                print('第',i,'次迭代后代价为:',cost)
    
    if plot:
        plt.plot(costs)
        plt.xlabel('iterations(per 100)')
        plt.ylabel('cost')
        plt.title('learning rate = '+ str(learning_rate))
        plt.show()

    return params

#预测
def predict(X, params, y):
    AL, _ = forward_propagation(X, params)
    y_pred = np.round(AL)
    predict_accuracy = y_pred[y_pred==y].size / y.size
    print('预测精度为: '+ str(round(predict_accuracy*100, 2))+ '%')

    return y_pred

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

layers_dims = [12288, 20, 7, 5, 1]
params = deep_nn(train_x, train_y, layers_dims, print_cost=True, plot=True)
print('训练集:')
train_y_pred =  predict(train_x, params, train_y)
print('测试集:')
test_y_pred =  predict(test_x, params, test_y)

#查看错误预测的样本
def print_mispredict_image(classes, X, y, y_pred):
    mis_idx = np.where(y_pred != y)[1]
    plt.rcParams['figure.figsize'] = (40,40)
    num_mis = len(mis_idx)

    for i in range(num_mis):
        idx = mis_idx[i]
        plt.subplot(1, num_mis, i+ 1)
        img = X[:,idx].reshape(64,64,3)
        plt.imshow(img)
        plt.title('prediction:'+ classes[int(y_pred[0,idx])].decode('utf-8') + '\nactual:'+ classes[int(y[0,idx])].decode('utf-8'))
    
    plt.show()

print_mispredict_image(classes, test_x, test_y, test_y_pred)