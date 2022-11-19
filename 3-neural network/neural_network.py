
from scipy.io import loadmat

data = loadmat("D:\Python\code/3-neural network\handwritten_digit_recognition(0~9).mat")
Weights = loadmat("D:\Python\code/3-neural network/nn_weights.mat")
X, y = data['X'], data['y']
w1, w2 = Weights['Theta1'], Weights['Theta2']

import numpy as np
def sigmoid(z):
    return 1/ (1 + np.exp(-z))

#正向传播
def feedforward(X, w1, w2):
    m = X.shape[0]
    A0 = np.c_[np.ones((m, 1)), X]
    Z1 = np.dot(A0, w1.T)
    A1 = np.c_[np.ones((m, 1)), sigmoid(Z1)]
    Z2 = np.dot(A1, w2.T)
    A2 = sigmoid(Z2)
    h = A2
    
    return h

def predict(X, w1, w2):
    h = feedforward(X, w1, w2)
    h_argmax = np.argmax(h, axis=1)
    y_pred = np.reshape(h_argmax + 1, (-1,1))
    
    return y_pred

'''实现'''
y_pred = predict(X, w1, w2)
accuracy = y_pred[y_pred==y].size / y.size

import matplotlib.pyplot as plt
plt.figure(figsize=(7,4))
plt.scatter(range(1, y.size + 1), y, marker='x', label='Actural')
plt.scatter(range(1, y.size + 1), y_pred, marker='*', label='Predict')
plt.title('accuracy {}% with Neural network'.format(round(accuracy*100, 2)))
plt.yticks(range(1, 11))
plt.legend()
plt.show()
