from planar_utils import *

X, Y = load_planar_dataset()

import matplotlib.pyplot as plt

#逻辑回归
from sklearn.linear_model import LogisticRegression
def logic_regression():
    clf = LogisticRegression()
    clf.fit(X.T, Y.T)
    
    plot_decision_boundary(lambda x:clf.predict(x), X, Y)
    plt.show()

    print('逻辑回归准确率: ' + str(clf.score(X.T, Y.T) * 100) +'%')

logic_regression()

'''1、定义结构:输入层单元，隐藏层单元，输出层单元'''
def layersize(X, Y, hidden_size=4):
    n_x = X.shape[0]
    n_h = hidden_size
    n_y = Y.shape[0]

    return n_x, n_h, n_y

'''2、初始化权重偏置'''
import numpy as np
def initialize_params(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y,1))

    params = {'W1': W1,
              'b1': b1,
              'W2': W2,
              'b2': b2}
    return params


'''3、前向传播'''
def forward_propagation(X, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {'Z1': Z1,
             'A1': A1,
             'Z2': Z2,
             'A2': A2}    
    
    return A2, cache 


'''4、计算代价'''
def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = 1/m * np.sum(-np.multiply(Y, np.log(A2)) - np.multiply((1 - Y), np.log(1 - A2)))
    
    return cost

'''5、反向传播'''
def back_propagation(params, cache, X, Y):
    A1 = cache['A1']
    A2 = cache['A2']
    W1 = params['W1']
    W2 = params['W2']

    m = Y.shape[1]
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    
    dZ1 = np.multiply(dA1, 1 - np.square(A1))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    grad = {'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2}
    
    return grad

'''6、更新参数'''
def update_params(params, grad, learning_rate=1):
    params['W1'] = params['W1'] - learning_rate * grad['dW1']
    params['b1'] = params['b1'] - learning_rate * grad['db1']
    params['W2'] = params['W2'] - learning_rate * grad['dW2']
    params['b2'] = params['b2'] - learning_rate * grad['db2']

    return params

'''7、整合流程'''
def nn_model_with_1_hidden_layer(X, Y, hidden_size, iterations, learning_rate=1, print_cost=False, plot_cost=False):
    n_x, n_h, n_y = layersize(X, Y, hidden_size)
    params = initialize_params(n_x, n_h, n_y)
    costs = []
    for i in range(iterations):
        A2, cache = forward_propagation(X, params)
        if i % 100 == 0:
            cost = compute_cost(A2, Y)
            costs.append(cost)
            if print_cost and i % 1000 == 0:
                print('第',i,'次迭代后代价为:',cost)
        grad = back_propagation(params, cache, X, Y)
        params = update_params(params, grad, learning_rate)
    
    if plot_cost:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations(per 100)')
        plt.title('cost with hidden_size={},learning rate={}'.format(hidden_size, learning_rate))
        plt.show()

    return params


'''8、预测'''
def predict(X, params):
    A2, cache = forward_propagation(X, params)
    y_pred = np.round(A2)

    return y_pred

datasets = load_extra_datasets()

#选择最佳隐层层单元数
def choose_best_nh(X, Y):
    #learning_rates = [0.001,0.003,0.01,0.03,0.1,0.3,1,3]
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50] 
    #for alpha in learning_rates:
    highest_accuracy = 0
    for n_h in hidden_layer_sizes:
        params = nn_model_with_1_hidden_layer(X, Y, n_h, 5000, 0.5)
        y_pred = predict(X, params)
        accuracy = y_pred[y_pred==Y].size / Y.size
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_nh = n_h
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.scatter(X[0,:],X[1,:],c=Y, cmap=plt.cm.Spectral)
        plt.subplot(1,2,2)
        plot_decision_boundary(lambda x:predict(x.T, params), X, Y)
        plt.title('accuracy: {}% with hidden size: {} and learning rate: {}'.format(round(accuracy * 100, 2), n_h, 0.5))
        plt.show()
        print('\n---------------------------------------------------\n')
        print('hidden size:',n_h,'learning rate:',0.5,'accuracy:',accuracy)
    
    print('highest_accuracy:'+ str(highest_accuracy)+ '\nbest_nh:'+ str(best_nh))

choose_best_nh(X, Y)