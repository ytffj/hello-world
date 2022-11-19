
'''参数初始化'''
import numpy as np
def initial_params(layer_size, initial_type='he'):
    L = len(layer_size)
    params = {}
    
    if initial_type == 'zeros':
        for l in range(1, L):
            params['W'+ str(l)] = np.zeros((layer_size[l], layer_size[l-1]))
            params['b'+ str(l)] = np.zeros((layer_size[l], 1))
    elif initial_type == 'random':
        for l in range(1, L):
            params['W'+ str(l)] = np.random.randn(layer_size[l], layer_size[l-1]) * 10
            params['b'+ str(l)] = np.zeros((layer_size[l], 1))
    else:
        for l in range(1, L):
            params['W'+ str(l)] = np.random.randn(layer_size[l], layer_size[l-1]) / np.sqrt(layer_size[l-1])
            params['b'+ str(l)] = np.zeros((layer_size[l], 1))
    
    return params



'''未正则化'''

def relu(z):
    s = np.maximum(0,z)
    
    return s
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def forward_propagation(X, params):     
    
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, A1, W1, Z2, A2, W2, Z3, A3, W3)
    
    return A3, cache

def compute_cost(A3, Y):

    m = Y.shape[1]

    cost = 1/m * np.nansum(-np.multiply(np.log(A3),Y) - np.multiply(np.log(1 - A3), 1 - Y))
    
    return cost

def backward_propagation(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, Z2, A2, W2, Z3, A3, W3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1/m * np.dot(dZ3, A2.T)
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, Z2 > 0)
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, Z1 > 0)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims = True)
    
    grads = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return grads

def update_parameters(params, grads, learning_rate):
    L = len(params) // 2 

    for l in range(1, L + 1):
        params["W" + str(l)] = params["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]
        
    return params

'''正则化'''

#L2正则化
def compute_cost_with_regularization(A3, Y, params, lambd):
    m = Y.shape[1]
    W1, W2, W3 = params['W1'], params['W2'], params['W3']

    cross_entropy = 1/m * np.nansum(-np.multiply(Y, np.log(A3)) - np.multiply(1-Y, np.log(1-A3)))
    reg_cost = lambd/(2*m) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    cost = cross_entropy  + reg_cost

    return cost

def back_propagation_with_regularization(X, Y, cache, lambd):
    m = Y.shape[1]
    Z1, A1, W1, Z2, A2, W2, Z3, A3, W3 = cache

    dZ3 = A3 - Y
    dW3 = 1/m * np.dot(dZ3, A2.T) + lambd/m * W3
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dZ2 = np.multiply(dA2, Z2 > 0)
    dW2 = 1/m * np.dot(dZ2, A1.T) + lambd/m * W2
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)

    dZ1 = np.multiply(dA1, Z1 > 0)
    dW1 = 1/m * np.dot(dZ1, X.T) + lambd/m * W1
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW3": dW3, "db3": db3,"dW2": dW2, "db2": db2,  "dW1": dW1, "db1": db1}

    return grads

#dropout
def forward_propagation_with_dropout(X, params, keep_prob):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']

    np.random.seed(1)
    
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1]) < keep_prob
    A1 = np.multiply(A1, D1) / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1]) < keep_prob
    A2 = np.multiply(A2, D2) / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, D1, W1, Z2, A2, D2, W2, Z3, A3, W3)

    return A3, cache


def back_propagation_with_dropout(X, Y, cache, keep_prob):
    m = Y.shape[1]
    Z1, A1, D1, W1, Z2, A2, D2, W2, Z3, A3, W3 = cache

    dZ3 = A3 - Y
    dW3 = 1/m * np.dot(dZ3, A2.T)
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3) * D2 / keep_prob

    dZ2 = np.multiply(dA2, Z2 > 0)
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2) * D1 / keep_prob

    dZ1 = np.multiply(dA1, Z1 > 0)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW3": dW3, "db3": db3,"dW2": dW2, "db2": db2,  "dW1": dW1, "db1": db1}

    return grads

'''抑梯度初始化和正则化'''
import matplotlib.pyplot as plt
def improved_deep_neural_network(X, Y, initial_type='zeros', keep_prob=1, lambd=0, iterations=30000, learning_rate=0.3, print_cost=False, plot=False):
    layer_size = [X.shape[0], 20, 3, Y.shape[0]]
    
    params = initial_params(layer_size, initial_type)
    
    costs = []
    for i in range(iterations):
        if keep_prob == 1:
            A3, cache = forward_propagation(X, params)
        else:
            A3, cache = forward_propagation_with_dropout(X, params, keep_prob)
        
        if i % 1000 == 0:
            if lambd == 0:
                cost = compute_cost(A3, Y)
            else:
                cost = compute_cost_with_regularization(A3, Y, params, lambd)
            costs.append(cost)
            if print_cost:
                print('第',i,'次迭代，代价为:',cost)

        if keep_prob == 1  and lambd == 0:
            grads = backward_propagation(X, Y, cache)
        elif keep_prob < 1:
            grads = back_propagation_with_dropout(X, Y, cache, keep_prob)
        else:
            grads = back_propagation_with_regularization(X, Y, cache, lambd)
        
        params = update_parameters(params, grads, learning_rate)

    if plot:
        plt.plot(costs)
        plt.xlabel('iterations(per 1000)')
        plt.ylabel('cost')
        plt.title('learning rate = {} keep_prob = {} lambda = {}'.format(learning_rate, keep_prob, lambd))
        plt.show()
    
    return params

#预测
def predict(X, Y, params):
    A3, _ = forward_propagation(X, params)
    y_pred = np.round(A3)
    accuracy = y_pred[y_pred==Y].size / Y.size 
    print('预测准确率为: '+ str(round(accuracy * 100, 2)) + '%')

    return y_pred


'''梯度检查'''
#参数字典转化为向量
def dictionary_to_vector(params):
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        
        vector = np.reshape(params[key], (-1,1))
        keys = keys + [key]*vector.shape[0]
        
        if count == 0:
            theta = vector
        else:
            theta = np.concatenate((theta, vector), axis=0)
        count = count + 1

    return theta, keys
#梯度转化为向量
def gradients_to_vector(grads):
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        vector = np.reshape(grads[key], (-1,1))
        
        if count == 0:
            theta = vector
        else:
            theta = np.concatenate((theta, vector), axis=0)
        count = count + 1

    return theta

#参数向量转化为字典
def vector_to_dictionary(theta):
    params = {}
    params["W1"] = theta[:20].reshape((5,4))
    params["b1"] = theta[20:25].reshape((5,1))
    params["W2"] = theta[25:40].reshape((3,5))
    params["b2"] = theta[40:43].reshape((3,1))
    params["W3"] = theta[43:46].reshape((1,3))
    params["b3"] = theta[46:47].reshape((1,1))

    return params
#正向传播计算J
def forward_propagation(X, Y, params): 
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    W3 = params['W3']
    b3 = params['b3']

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    m = Y.shape[1]
    cost = (1 / m) * np.sum(-np.multiply(np.log(A3), Y) - np.multiply(np.log(1 - A3), 1 - Y))

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache

def gradient_check(params, grads, X, Y, epsilon=1e-7):
    params_vector, _ = dictionary_to_vector(params)
    grad_vector = gradients_to_vector(grads)
    num_params = params_vector.shape[0]
    grad_approx = np.zeros((num_params, 1))
    
    for i in range(num_params):
        params_plus = params_vector.copy()
        params_plus[i][0] += epsilon
        params_minus = params_vector.copy()
        params_minus[i][0] -= epsilon
        params_plus_to_dict = vector_to_dictionary(params_plus)
        params_minus_to_dict = vector_to_dictionary(params_minus)

        J_plus,  _ = forward_propagation(X, Y, params_plus_to_dict)
        J_minus, _ = forward_propagation(X, Y, params_minus_to_dict)

        grad_approx_i = (J_plus  - J_minus) / (2 * epsilon)

        grad_approx[i][0] = grad_approx_i
    
    dist = np.linalg.norm(grad_approx - grad_vector)
    addition = np.linalg.norm(grad_approx) + np.linalg.norm(grad_vector)

    diff = dist / addition

    if diff < 1e-7:
        print('梯度检查pass')
    else:
        print('梯度异常')

    return diff