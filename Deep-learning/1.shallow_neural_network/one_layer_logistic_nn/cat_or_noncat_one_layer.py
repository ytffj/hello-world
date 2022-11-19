from lr_utils import load_dataset
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()


#处理形状
train_set_x_orig_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_orig_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

#归一化
train_x = train_set_x_orig_flatten /  255 
test_x = test_set_x_orig_flatten / 255

#sigmoid
import numpy as np
def sigmoid(z):
    
    return 1 / (1 + np.exp(-z))

#初始化权重
def initialize_params(X):
    w = np.zeros((X.shape[0], 1))
    b = 0

    return w, b

#正反向传播
def propagation(X, w, b, Y):
    Z = np.dot(w.T, X)
    A = sigmoid(Z)

    m = Y.shape[1]
    cost = 1/m * np.sum(np.multiply(-Y, np.log(A)) - np.multiply(1-Y, np.log(1-A)))

    dZ = A - Y
    dw = 1/m * np.dot(X, dZ.T)
    db = 1/m * np.sum(dZ)

    return cost,  dw, db

#梯度下降
import matplotlib.pyplot as plt
def gradientdescent(X, w, b, Y, learning_rate=1, iterations=2000, print_cost=False, plot=False):
    costs = []
    for i in range(iterations):
        cost, dw, db = propagation(X, w, b, Y)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost and i % 1000 == 0:
                print("第",i,'次迭代后代价为:',cost)
    
    if plot:
        plt.plot(costs)
        plt.xlabel('iterations(per 1000)')
        plt.ylabel('cost')
        plt.title('cost with learning rate = '+ str(learning_rate))
        plt.show()

    return w , b , costs

#预测
def predict(w, b, x):
    y_pred = np.round(sigmoid(np.dot(w.T, x) + b))

    return y_pred

learning_rates = [0.01 , 0.001 , 0.0001]
for learning_rate in learning_rates:
    w, b = initialize_params(train_x)
    w , b , costs = gradientdescent(train_x, w, b, train_set_y_orig, learning_rate)
    train_predict = predict(w, b, train_x)
    test_predict = predict(w, b, test_x)
    train_accuracy = train_predict[train_predict==train_set_y_orig].size / train_set_y_orig.size
    test_accuracy = test_predict[test_predict==test_set_y_orig].size / test_set_y_orig.size
    print('\n'+'----------------------------'+'\n')
    print('learning rate:',learning_rate)
    print('train accuracy: {}%'.format(round(train_accuracy * 100, 2)))
    print('test accuracy: {}%'.format(round(test_accuracy * 100, 2)))
    plt.plot(costs, label=str(learning_rate))

plt.ylabel('cost')
plt.xlabel('iteration(per 1000)')
plt.title('cost with learning rate')
plt.legend()
plt.show()