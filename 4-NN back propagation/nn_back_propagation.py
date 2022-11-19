from scipy.io import loadmat

data = loadmat('D:\Python\code/4-NN back propagation\handwritten_digit_recognition(0~9).mat')
X , y_orgin = data['X'] , data['y']

import numpy as np
classes = np.unique(y_orgin).size
y_onehot  = np.zeros((y_orgin.shape[0], classes))
y_onehot[np.arange(y_orgin.shape[0]).reshape(-1,1), y_orgin - 1]  = 1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#初始化权重参数
def randominitilizeweight(X , hidden_size , y):
    input_size = X.shape[1]
    output_size = y.shape[1]

    epsilon_init1 = np.sqrt(6 / (input_size + hidden_size))
    epsilon_init2 = np.sqrt(6 / (hidden_size + output_size))

    w1 = np.random.random(hidden_size * (input_size + 1)) * 2 * epsilon_init1 - epsilon_init1
    w2 = np.random.random(output_size * (hidden_size + 1)) * 2 * epsilon_init2 - epsilon_init2

    W = np.concatenate((w1, w2))

    return W

#正向传播
def forwardpropagation(X, w1, w2):
    m = X.shape[0]
    a0 = np.c_[np.ones((m, 1)), X]
    z1 = np.dot(a0, w1.T)
    a1 = np.c_[np.ones((m, 1)), sigmoid(z1)]
    z2 = np.dot(a1, w2.T)
    a2 = sigmoid(z2)
    h = a2

    return  a0, a1, h

#代价
def cost(W, X, y, hidden_size, lambd):
    input_size , output_size = X.shape[1] , y.shape[1]
    w1 = W[:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)
    w2 = W[hidden_size * (input_size + 1):].reshape(output_size, hidden_size+ 1)
    a0, a1, h = forwardpropagation(X, w1, w2)

    m = X.shape[0]
    cross_entropy = 1/m * np.sum(np.multiply(-y, np.log(h)) - np.multiply(1-y, np.log(1-h)))
    reg_cost = lambd/(2*m) * (np.sum(np.square(w1[:,1:])) + np.sum(np.square(w2[:,1:])))
    cost = cross_entropy + reg_cost
    
    return cost

#反向传播
def backpropagation(W, X, y, hidden_size, lambd):
    input_size , output_size = X.shape[1] , y.shape[1]
    w1 = W[:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)
    w2 = W[hidden_size * (input_size + 1):].reshape(output_size, hidden_size+ 1)
    a0, a1, h = forwardpropagation(X, w1, w2)
    
    m = X.shape[0]
    dz2 = h - y
    dw2 = np.dot(dz2.T, a1) / m
    da1 = np.dot(dz2, w2)

    dz1 = da1 * a1 * (1 - a1)
    dz1 = dz1[:,1:]
    
    dw1 = np.dot(dz1.T, a0) / m
    
    reg_dw2 = lambd/m * w2[:,1:]
    reg_dw1 = lambd/m * w1[:,1:]
    
    dw1[:,1:] += reg_dw1
    dw2[:,1:] += reg_dw2
    dw = np.concatenate((dw1.ravel(), dw2.ravel()))
    
    return dw

#优化
from scipy.optimize import minimize
def optimize(X, hidden_size, y, lambd):
    initial_W = randominitilizeweight(X, hidden_size, y)
    res = minimize(fun=cost, x0=initial_W, args=(X, y, hidden_size, lambd), method='TNC', jac=backpropagation)
    W =  res.x

    return W
#预测
def predict(x, W, hidden_size, train_X, train_y):
    input_size , output_size = train_X.shape[1] , train_y.shape[1]
    w1 = W[:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)
    w2 = W[hidden_size * (input_size + 1):].reshape(output_size, hidden_size+ 1)
    a0, a1, h = forwardpropagation(x, w1, w2)
    h_argmax = np.argmax(h, axis=1)
    y_pred = np.reshape(h_argmax + 1, (-1, 1))

    return y_pred

#不同正则强度
import matplotlib.pyplot as plt
def plot(X, hidden_size, y, lambdas):
    for l in lambdas:
        trained_theta  = optimize(X, hidden_size, y, l)

        y_pred = predict(X, trained_theta, hidden_size, X, y)
        accuracy = y_pred[y_pred==y_orgin].size / y_orgin.size

        plt.figure(figsize=(16, 8))
        plt.scatter(range(1, X.shape[0] + 1), y_orgin, marker='x', label='Actual')
        plt.scatter(range(1, X.shape[0] + 1), y_pred, marker='*', label='Predict')
        plt.legend()
        plt.title('Train accuracy {}% with lambda = {}'.format(round(accuracy * 100 , 2), l))
        plt.show()
        
lambdas = [0,0.1,1,10,100]
plot(X, 25, y_onehot, lambdas)            
