from scipy.io import loadmat

data = loadmat('D:\Python\code/5-bias vs variance\data.mat')
X, y ,Xtest, ytest, Xval, yval = data['X'], data['y'], data['Xtest'], data['ytest'], data['Xval'], data['yval']

#数据可视化
import matplotlib.pyplot as plt 
def datavisiblize(X_train, y_train):
    plt.scatter(X_train, y_train)
    plt.xlabel('Change in water level')
    plt.ylabel('Water flowing out of the dam')
    plt.show()

#特征映射
from sklearn.preprocessing import PolynomialFeatures
def feature_mapping(x, degree):
    poly = PolynomialFeatures(degree)
    X = poly.fit_transform(x)

    return X

#均值归一
import numpy as np
def normalization(X):
    X[:,1:] = (X[:,1:] - np.mean(X[:,1:], axis=0)) / np.std(X[:,1:], axis=0)

    return X

#初始化权重
def initializetheta(X):
    np.random.seed(1)
    initialtheta = np.random.randn(X.shape[1])

    return initialtheta

#计算代价
def cost(theta, X, y, lambd):
    m = X.shape[0]
    h = X.dot(theta.reshape(-1,1))
    ms_error = 1/(2*m) * np.sum(np.square(h - y))
    reg_error = lambd/(2*m) * np.sum(np.square(theta[1:]))
    cost = ms_error + reg_error

    return cost

#计算梯度
def grad(theta, X, y, lambd):
    m = X.shape[0]
    h = X.dot(theta.reshape(-1,1))
    grad = 1/m * X.T.dot(h - y)
    reg_grad =  lambd/m * theta[1:]
    grad[1:] += reg_grad.reshape(-1,1)
    grad.shape = theta.shape

    return grad

#建立训练模型
from scipy.optimize import minimize
def train(X, y, lambd):
    initialtheta = initializetheta(X)
    res = minimize(fun=cost, x0=initialtheta, args=(X, y, lambd), method='TNC', jac=grad)
    return res

#用不同参数进行训练， 比较验证集代价，找到最佳多项式次数 和 正则化参数
def find_best_degree_and_lambd(x, y, Xval, yval):
    train_costs = []
    cv_costs = []
    for degree in range(1,20):
        X = feature_mapping(x, degree)
        X_norm = normalization(X)
        res = train(X_norm, y, 0)
        trained_cost =  res.fun
        train_costs.append(trained_cost)
        trained_theta = res.x

        X_val = feature_mapping(Xval, degree)
        X_val_norm = normalization(X_val)
        cv_cost = cost(trained_theta, X_val_norm, yval, 0)
        cv_costs.append(cv_cost)

    best_degree = np.argmin(cv_costs) + 1
    plt.plot(range(1,20),train_costs, label='train set cost')
    plt.plot(range(1,20),cv_costs, label='cv set cost')
    plt.xticks(ticks=range(1,20))
    plt.xlabel('polynomial degree')
    plt.ylabel('cost')
    plt.title('cost with different polynomial degree')
    plt.legend()
    plt.show()

    train_costs = []
    cv_costs = []
    lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    for lambd in lambdas:
        X = feature_mapping(x, best_degree)
        X_norm = normalization(X)
        res = train(X_norm, y, lambd)
        trained_cost =  res.fun
        train_costs.append(trained_cost)
        trained_theta = res.x

        X_val = feature_mapping(Xval, best_degree)
        X_val_norm = normalization(X_val)
        cv_cost = cost(trained_theta, X_val_norm, yval, 0)
        cv_costs.append(cv_cost)
    
    best_lambda = lambdas[np.argmin(cv_costs)]
    plt.plot(lambdas, train_costs, label='train set cost')
    plt.plot(lambdas, cv_costs, label='cv set cost')
    plt.xlabel('lambda')
    plt.ylabel('cost')
    plt.title('cost with different lambda')
    plt.legend()
    plt.show()

    print('best degree:',best_degree)
    print('best lambda:',best_lambda)
    return best_degree, best_lambda

best_degree, best_lambda = find_best_degree_and_lambd(X, y, Xval, yval)

#用最佳多项式次数和正则参数在测试集上检查效果
def train_with_best_degree_and_lambda(x, y, Xtest, ytest, best_degree, best_lambda):
    X = feature_mapping(x, best_degree)
    X_norm = normalization(X)
    res = train(X_norm, y, best_lambda)

    train_cost = res.fun
    trained_theta = res.x

    X_test = feature_mapping(Xtest, best_degree)
    X_test_norm = normalization(X_test)
    test_cost = cost(trained_theta, X_test_norm, ytest, 0)
    
    print('train set cost:',train_cost)
    print('test set cost:',test_cost)

    x_scale = np.linspace(Xtest.min()- 1, Xtest.max() + 1).reshape(-1,1)
    X_scale = feature_mapping(x_scale, best_degree)
    X_scale_norm = normalization(X_scale)
    y_pred = X_scale_norm.dot(trained_theta.reshape(-1,1))
   
    plt.scatter(Xtest, ytest,  label='Actual')
    plt.plot(x_scale, y_pred, label='Predict')
    plt.title('test set prediction with degress = {} lambda = {}'.format(best_degree,best_lambda))
    plt.legend()
    plt.show()

    return  trained_theta

trained_theta = train_with_best_degree_and_lambda(X, y, Xtest, ytest, best_degree, best_lambda)

#用训练好的参数进行预测

def predict(x, X_train, trained_theta, best_degree):
    X_train = feature_mapping(X_train, best_degree)
    mean , std = np.mean(X_train[:,1:], axis=0), np.std(X_train[:,1:], axis=0)
    X = feature_mapping(x, best_degree)
    X[:,1:] = (X[:,1:] - mean) / std
    y_pred = X.dot(trained_theta.reshape(-1,1))

    return y_pred

