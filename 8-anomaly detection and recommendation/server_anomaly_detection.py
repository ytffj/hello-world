from scipy.io import loadmat

data = loadmat('D:\Python\code\8-anomaly detection and recommendation\data\server-throughput-and-latency.mat')

X, Xval, yval = data['X'], data['Xval'], data['yval']

import matplotlib.pyplot as plt
def data_visialize(X):
    plt.scatter(X[:,0], X[:,1])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()

#计算高斯参数
import numpy as np
def compute_gaussian_params(X):
    μ = np.mean(X, axis=0)
    sigma2 = np.mean((X - μ) ** 2, axis=0)
    #sigma2 = np.var(X, axis=0)
    return μ, sigma2

#计算高斯分布概率密度
def compute_gaussian_prob(X, μ, sigma2):
    m, n = X.shape
    P_X = np.ones((m, 1))
    for j in range(n):
        P_Xj = 1/np.sqrt(2 * np.pi * float(sigma2[j])) * np.exp(- np.square(X[:,j] -  float(μ[j]))/(2 * float(sigma2[j])))
        P_X *= np.reshape(P_Xj, (-1,1))
    
    return P_X

#选择最优概率密度阈值
def choose_best_epsilon(P_X, y):
    best_F1 = 0
    for epsilon in np.linspace(P_X.min(), P_X.max(), 1000):
        y_pred = P_X < epsilon
        tp = np.count_nonzero(np.logical_and(y_pred==1, y==1))
        fp = np.count_nonzero(np.logical_and(y_pred==1, y==0))
        fn = np.count_nonzero(np.logical_and(y_pred==0, y==1))

        if tp + fp == 0: continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2 * prec * rec / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_ε = epsilon
        
    return best_F1, best_ε

μ, sigma2 = compute_gaussian_params(Xval[np.where( yval==0 )[0]])
p = compute_gaussian_prob(X, μ, sigma2)
pval = compute_gaussian_prob(Xval, μ, sigma2)

'''scipy统计'''
from scipy import stats

#p = np.multiply(stats.norm(μ[0], sigma2[0]).pdf(X[:,0]).reshape(-1,1) , stats.norm(μ[1], sigma2[1]).pdf(X[:,1]).reshape(-1,1))
#pval = np.multiply(stats.norm(μ[0], sigma2[0]).pdf(Xval[:,0]).reshape(-1,1) , stats.norm(μ[1], sigma2[1]).pdf(Xval[:,1]).reshape(-1,1))

best_F1, best_ε = choose_best_epsilon(pval, yval)
print('best_F1:',best_F1)
print('best_ε:',best_ε)
X_abnormal = X[np.where( p < best_ε )[0]]
Xval_abnormal = Xval[np.where( pval < best_ε )[0]]

#在验证集上的表现
fig,  ax = plt.subplots(1,2)
ax[0].scatter(Xval[:,0],Xval[:,1])
ax[0].scatter(Xval[np.where( yval==1 )[0],0],Xval[np.where( yval==1 )[0],1], marker='X', c='r')
ax[0].set_title('Actual')

ax[1].scatter(Xval[:,0],Xval[:,1])
ax[1].scatter(Xval_abnormal[:,0],  Xval_abnormal[:,1], marker='X', c='r')
ax[1].set_title('Predict')
plt.show()

#在测试集上的表现
plt.scatter(X[:,0],X[:,1])
plt.scatter(X_abnormal[:,0],  X_abnormal[:,1], marker='X', c='r')
plt.title('test set Predict')
plt.show()