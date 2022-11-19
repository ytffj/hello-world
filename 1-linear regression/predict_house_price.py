import numpy as np

#提取数据
data = np.loadtxt("D:\Python\code/1-linear regression\house_price_with_housesize_and_bedroomnumber.txt",delimiter=',')
m, n = data.shape
x, y = data[:,:n-1], data[:,n-1].reshape(-1,1)

import matplotlib.pyplot as plt

#可视化数据
def dataVisiablize(x, y):
    plt.scatter(x[:,0],y)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('POPULATION AND PROFIT')
    plt.show()

#代价函数
def compute_cost(X, theta, y):
    cost = 1/(2*m) * np.sum(np.square(np.dot(X, theta) - y))

    return cost

#梯度函数
def compute_grad(X, theta, y):
    grad = (1/m) * np.dot(X.T, np.dot(X, theta) - y)

    return grad

#梯度下降
def gradient_Descent(X, theta, y, iteration, alpha):
    costs = []
    for i in range(iteration):
        grad = compute_grad(X, theta, y)
        theta = theta - alpha * grad
        if i % 5 == 0:
            cost = compute_cost(X, theta, y)
            costs.append(cost)

    return theta, costs

#正规方程
def normalEqn(X, y):
    theta = ((np.linalg.inv(X.T.dot(X))).dot(X.T)).dot(y)

    return theta

#预测
def predict(x, theta):
    m = x.shape[0]
    x_norm = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    X = np.c_[np.ones((m,1)), x_norm]
    prediction = np.dot(X, theta)

    return prediction

'''实现'''
#均值归一
x_norm = (x - np.mean(x, axis=0))/ np.std(x, axis=0)

#初始化参数
X = np.c_[np.ones((m,1)), x_norm]
np.random.seed(1)
theta = np.random.randn(n,1)

iteration = 100
alpha = 0.1
theta, costs = gradient_Descent(X, theta, y, iteration, alpha)
theta_Eqn = normalEqn(X, y)
prediction_gradient = predict(x, theta)
prediction_normalEqn = predict(x, theta_Eqn)

from sklearn.linear_model import LinearRegression
sk_model = LinearRegression().fit(x, y)
score = sk_model.score(x, y)
prediction_sklearn = sk_model.predict(x)

fig, ax = plt.subplots(1,2,figsize=(16,8))

ax[0].set_title('cost with iteration')
ax[0].plot(range(0,iteration,5), costs)

ax[1].set_title('prediction with gradientDescent/normalEqn/sk_model')
ax[1].scatter(x[:,0], y, marker='o', label='Actural' )
ax[1].scatter(x[:,0], prediction_gradient, marker='x', label='gradientDescent')
ax[1].scatter(x[:,0], prediction_normalEqn, marker='+', label='normalEqn')
ax[1].scatter(x[:,0], prediction_sklearn, marker='*', label='sk_model')

plt.legend()
plt.show()

