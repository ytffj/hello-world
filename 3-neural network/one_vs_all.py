from scipy.io import loadmat

data = loadmat('D:\Python\code/3-neural network\handwritten_digit_recognition(0~9).mat')
x , y = data['X'], data['y']

import numpy as np
X = np.c_[np.ones((x.shape[0], 1)), x]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#代价
def cost(theta, X, y, lambd):
    m = X.shape[0]
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    cross_entropy = 1/m * np.sum(np.multiply(-y, np.log(h)) - np.multiply(1-y, np.log(1-h)))
    reg_cost = lambd / (2*m) * np.sum(np.square(theta[1:]))
    cost = cross_entropy + reg_cost

    return cost
#梯度
def gradient(theta, X, y, lambd):
    m = X.shape[0]
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    grad = 1/m * (X.T.dot(h - y))
    reg_grad = lambd/m * theta[1:]
    grad[1:] += reg_grad.reshape(-1,1)

    return grad.ravel()

from scipy.optimize import minimize

#one_vs_all
def one_vs_all(X, y, lambd):
    k = np.unique(y).size
    num_params = X.shape[1]
    thetas = np.zeros((k, num_params))
    for i in range(k):
        y_i = np.where(y==(i if i !=0 else 10), 1, 0)
        theta = thetas[i] 
        res = minimize(fun=cost, x0=theta, args=(X, y_i, lambd), jac=gradient, method='TNC')
        thetas[i] = res.x
    
    return thetas

def predict(x, thetas):
    m  = x.shape[0]
    X = np.c_[np.ones((m, 1)), x]
    h = sigmoid(X.dot(thetas.T))
    y_pred = np.argmax(h, axis=1).reshape(-1,1)

    return y_pred

lambdas = [0, 0.1, 1, 10, 100]
import matplotlib.pyplot as plt

#不同正则强度结果
def plot_with_different_lambda(X, y, lambdas):
    for l in lambdas:
        thetas = one_vs_all(X, y, l)
        y_pred = predict(x, thetas)
        accuracy = y_pred[y_pred==np.where(y==10, 0, y)].size / y.size

        plt.figure(figsize=(7, 4))
        plt.scatter(range(1, X.shape[0] + 1), np.where(y==10, 0, y), marker='x', label='Actural')
        plt.scatter(range(1, X.shape[0] + 1), y_pred, marker='*', label='Predict')
        plt.yticks(ticks=range(10), labels=range(10))
        plt.xlabel('picture')
        plt.ylabel('number')
        plt.title('Train accuracy {}% with lambda = {}'.format(round(accuracy * 100, 2), l))
        plt.legend(loc='upper right')
        plt.show()

plot_with_different_lambda(X, y, lambdas)
