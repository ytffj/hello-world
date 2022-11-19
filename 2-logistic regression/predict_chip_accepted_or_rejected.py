import numpy as np

data = np.loadtxt('D:\Python\code/2-logistic regression\chip_accepted_or_rejected_with_two_test_result.txt', delimiter=',')

import matplotlib.pyplot as plt

#数据可视化
def Visiblizedata(data):
    pos_data = data[data[:,-1]==1]
    neg_data = data[data[:,-1]==0]
    plt.scatter(pos_data[:,0], pos_data[:,1], c='g', marker='o', label='Admitted')
    plt.scatter(neg_data[:,0], neg_data[:,1], c='r', marker='x', label='Not admitted')
    plt.xlabel('Test1')
    plt.ylabel('Test2')
    plt.legend(loc='upper right')
    #plt.show()

#构造多项式特征
def polynomial_feature(x, degree):
    x1, x2 = x[:,0], x[:,1]
    m = x.shape[0]
    X = np.zeros((m, int((1 + degree + 1) * (1 + degree)/ 2)))
    k = 0
    for i in range(degree+1):
        for j in range(i+1):
            X[:,k] = np.power(x1, j)  * np.power(x2, i - j)
            k += 1
    
    return X

from  sklearn.preprocessing import PolynomialFeatures

def Polynomial_Features(x, degree):
    poly = PolynomialFeatures(degree)
    X = poly.fit_transform(x)
    
    return X


#sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#代价函数
def compute_cost(theta, X, y, lambd):
    theta = theta.reshape(-1,1)
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    cross_entropy = (-1/m) * np.sum(np.multiply(y, np.log(h)) + np.multiply((1 - y), np.log(1 - h)))
    reg_cost = lambd / (2 * m) * np.sum(np.square(theta[1:]))

    return cross_entropy + reg_cost

#梯度函数
def compute_grad(theta, X, y, lambd):
    theta = theta.reshape(-1,1)
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    cross_grad = (1/m) * np.dot(X.T, (h - y))
    reg_grad = lambd / m *  theta[1:]
    grad = cross_grad + np.r_[[[0]], reg_grad]
    grad.shape = X.shape[1]
    return grad

#梯度下降
def gradient_Descent(X, theta, y, alpha=1, lambd=1, iteration=10000, print_cost=False):
    costs = []
    for i in range(iteration):
        grad = compute_grad(theta, X, y, lambd)
        theta = theta - alpha * grad
        if i % 10 == 0:
            cost = compute_cost(theta, X, y, lambd)
            costs.append(cost)
            if print_cost and i % 100 == 0:
                print(cost)

    return theta, costs

'''实现'''
x, y = data[:, :-1], data[:, -1].reshape(-1,1)
degree = 6
X = polynomial_feature(x, degree)

np.random.seed(1)
theta = np.random.randn(X.shape[1])

import  scipy.optimize as opt
from sklearn.linear_model import LogisticRegression

lambdas =  [0.1,1,10,100]
def optimize(X, y, theta, method='gradient_descent'):
    for l in lambdas:
        if method == 'sc_minimize':
            res = opt.minimize(fun=compute_cost, x0=theta, args=(X, y, l), jac=compute_grad)
            trained_theta = res.x.reshape(-1,1)
        elif method == 'sk_LR':
            model = LogisticRegression(penalty='l2', C=1/l).fit(X, np.ravel(y))
        elif method == 'gradient_descent':
            trained_theta, costs = gradient_Descent(X, theta, y,  0.1, lambd=l)
            trained_theta = trained_theta.reshape(-1,1)
        
        if method == 'sk_LR':
            accuracy = model.score(X, y)
        else:
            y_pred = np.round(sigmoid(np.dot(X, trained_theta)))
            accuracy = y_pred[y_pred==y].size / y.size

        x1_min, x1_max = x[:,0].min() , x[:,0].max()
        x2_min, x2_max = x[:,1].min() , x[:,1].max()
        x1 , x2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
        x_grid = np.c_[x1.ravel(), x2.ravel()]
        x_grid_poly = polynomial_feature(x_grid, degree)
        if method == 'sk_LR':
            h = model.predict(x_grid_poly)
        else:
            h = sigmoid(np.dot(x_grid_poly, trained_theta))
        h.shape = x1.shape

        Visiblizedata(data)
        plt.contour(x1, x2, h, [0.5])
        plt.title('train set accuracy = {} % with lambda = {}'.format((round(accuracy * 100, 2)) , l))
        plt.show()


optimize(X, y, theta,'sc_minimize')

QA = {0: 'rejected', 1: 'accepted'}
def predict(x, trained_theta):
    X = polynomial_feature(x, degree)
    h = sigmoid(np.dot(X, trained_theta))
    y_pred = np.round(h)

    print('预测为：',QA[int(y_pred)])

