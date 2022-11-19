from scipy.io import loadmat

data = loadmat('D:\Python\code/6-SVM\data\data1.mat')
X , y = data['X'], data['y']

import matplotlib.pyplot as plt
import numpy as np

def datavisiablize(X, y):
    pos_data = X[np.where(y==1)[0]]
    neg_data = X[np.where(y==0)[0]]
    plt.scatter(pos_data[:,0], pos_data[:,1], marker='o')
    plt.scatter(neg_data[:,0], neg_data[:,1], marker='x')
    #plt.show()

from sklearn import svm

'''线性分类'''
svc = svm.LinearSVC(C=1, loss='hinge')
svc.fit(X, y.ravel())
score = svc.score(X, y.ravel())

#样本预测的置信度
SVM_confidence = svc.decision_function(X)
plt.figure(figsize=(12,8))
plt.scatter(X[:,0], X[:,1], c=SVM_confidence, cmap=plt.cm.Spectral)
plt.show()

#高斯核
def gaussian_kernel(x1, x2, sigma):
    return  np.exp(-np.sum((x1-x2)**2)/ (2 * sigma ** 2))

#决策界限
def plot_Decision_boundary(svc, x , y):
    x1_min , x1_max = x[:,0].min() , x[:,0].max()
    x2_min , x2_max = x[:,1].min() , x[:,1].max()
    xx1 , xx2 = np.meshgrid(np.linspace(x1_min , x1_max), np.linspace(x2_min , x2_max))
    Z = svc.predict(np.c_[xx1.ravel() , xx2.ravel()])
    Z.shape = xx1.shape

    datavisiablize(x , y)
    plt.contour(xx1, xx2, Z)
    plt.show()

'''非线性分类'''
data2 = loadmat('D:\Python\code/6-SVM\data\data2.mat')
X, y = data2['X'] , data2['y']
svc = svm.SVC(C=100, gamma=10, probability=True)
svc.fit(X, y.ravel())
score = svc.score(X, y.ravel())
probability = svc.predict_proba(X)[:,0]
plt.scatter(X[:,0], X[:,1], s=30, c=probability, cmap=plt.cm.Spectral)
plt.show()

#最佳C和gamma
def find_best_params(X, y, Xval, yval):
    C_values = [0.1, 0.3, 1, 3, 10, 30, 100]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

    highest_score = 0
    best_C, best_gamma = None, None
    for c in C_values:
        for g in gamma_values:
            print('当前组合: c={}, gamma={}'.format(c, g))
            svc = svm.SVC(C=c, gamma=g)
            svc.fit(X, y.ravel())
            score = svc.score(Xval, yval.ravel())
            print('分数:', round(score*100, 2))
            if score > highest_score:
                highest_score = score
                best_C, best_gamma =  c, g

    return best_C, best_gamma

'''垃圾邮件分类'''
spamTrain = loadmat("D:\Python\code/6-SVM\data\spamTrain.mat")
spamTest = loadmat('D:\Python\code/6-SVM\data\spamTest.mat')
X, y = spamTrain['X'], spamTrain['y']
Xtest, ytest = spamTest['Xtest'], spamTest['ytest']
train_size = int(X.shape[0] *  0.7)

best_C, best_gamma = find_best_params(X[:train_size], y[:train_size], X[train_size:], y[train_size:])
svc = svm.SVC(C=best_C, gamma=best_gamma)
svc.fit(X, y.ravel())
print('训练集精度: {}%'.format(round(svc.score(X, y) * 100, 2)))
print('测试集精度: {}%'.format(round(svc.score(Xtest, ytest) * 100, 2)))
