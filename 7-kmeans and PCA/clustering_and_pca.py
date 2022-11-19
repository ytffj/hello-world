from scipy.io import loadmat

data = loadmat('D:\Python\code/7-kmeans and PCA\data\data2.mat')
X = data['X']

import matplotlib.pyplot as plt
def datavisiablize(X):
    plt.scatter(X[:,0], X[:,1])
    plt.show()

#随机初始化聚类中心
import numpy as np
def random_initialize_centroids(X, K):
    random_X = np.random.permutation(X)
    centroids = random_X[:K]

    return centroids

#将数据归属到最近的聚类中心
def find_closest_centroids(X, centroids):
    m ,K = X.shape[0], centroids.shape[0]
    dist_sq =  np.zeros((m, K))
    for k in range(K):
        centroid = centroids[k]
        dist_sq_k = np.sum(np.square(X - centroid), axis=1)
        dist_sq[:,k] = dist_sq_k

    X_to_c_idx = np.argmin(dist_sq, axis=1)

    return X_to_c_idx

#将聚类中心移动到归属点的中心
def move_to_center(X, X_to_c_idx, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        cluster_k = X[np.argwhere(X_to_c_idx==k)]
        centroids[k] = np.mean(cluster_k, axis=0)
    
    return centroids

#计算代价
def compute_cost(X, X_to_c_idx, centroids):
    m = X.shape[0]
    cost = 1/m  * np.sum(np.square(X - centroids[X_to_c_idx]))

    return cost

'''实现'''
def K_means(X, K, random_times=10, plot=False):
    #多次随机初始化聚类中心避免局部最优
    mincost = float('inf')
    for _ in range(random_times):
        centroids = random_initialize_centroids(X, K)
        pre_centroids = np.zeros_like(centroids)
        costs_hist = []
        clusters_hist, centroids_path = {}, {}
        
        for k in range(K):
            centroids_path['centroid'+str(k+1)] = centroids[k].reshape(1,-1)

        while not (centroids==pre_centroids).all():

            pre_centroids = centroids.copy()
            X_to_c_idx = find_closest_centroids(X, centroids)
            centroids = move_to_center(X, X_to_c_idx, K)

            cost = compute_cost(X, X_to_c_idx, centroids)
            costs_hist.append(cost)
            #记录簇类分配和簇类中心移动
            for k in range(K):
                clusters_hist['cluster'+str(k+1)] = X[np.squeeze(np.argwhere(X_to_c_idx==k))]
                centroids_path['centroid'+str(k+1)] = np.concatenate((centroids_path['centroid'+str(k+1)], centroids[k].reshape(1,-1)),axis=0)

        if cost < mincost:
            mincost = cost
            best_X_to_c_idx = X_to_c_idx
            best_centroids  = centroids
            best_clusters_hist = clusters_hist
            best_centroids_path = centroids_path

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(12, 8))
        for k in range(K):
            cluster_k = best_clusters_hist['cluster'+str(k+1)]
            centroid_k = best_centroids_path['centroid'+str(k+1)]
            ax[0].scatter(cluster_k[:,0], cluster_k[:,1], label='cluster'+str(k+1))
            ax[0].plot(centroid_k[:,0], centroid_k[:,1], label='centroid'+str(k+1))
            ax[0].scatter(centroid_k[:,0], centroid_k[:,1], marker='X', label='centroid'+str(k+1))

        ax[0].legend()
        ax[1].plot(costs_hist,label='cost')
        ax[1].legend()
        plt.show()

        
    return mincost, best_X_to_c_idx, best_centroids

'''查看聚类效果'''
K_means(X, 3, plot=True)

'''图片压缩'''
from   matplotlib import image  as mpimg

image = mpimg.imread('D:\Python\code/7-kmeans and PCA\data\Bird_small.png')        
img_data = loadmat('D:\Python\code/7-kmeans and PCA\data\Bird_small.mat')
A = img_data['A']
A = A / 255.
X = np.reshape(A,(A.shape[0] * A.shape[1], A.shape[2]))

#手动图片压缩
mincost, best_X_to_c_idx, best_centroids = K_means(X, 16)
X_dim = best_centroids[best_X_to_c_idx]
A_dim = np.reshape(X_dim, A.shape)

#sklearn图片压缩
from  sklearn.cluster import KMeans
model = KMeans(n_clusters=16, n_init=10)
model.fit(X) 
centroids = model.cluster_centers_
idx = model.predict(X)
X_ = centroids[idx]
A_ = np.reshape(X_, A.shape)

fig,ax = plt.subplots(1,3)
ax[0].set_title('original image')
ax[0].imshow(A)
ax[1].set_title('manual compressed')
ax[1].imshow(A_dim)
ax[2].set_title('sk compressed')
ax[2].imshow(A_)
plt.show()

data = loadmat('D:\Python\code/7-kmeans and PCA\data\data1.mat')
X = data['X']

#pca得到主成分矩阵
def pca(X):
    #特征归一
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    #协方差
    X_cov = 1/X.shape[0] * np.dot(X.T, X)
    #奇异值分解
    U, S, V = np.linalg.svd(X_cov)
    
    return U, S, V

#根据保留特征信息量选择主成分数量k
def choose_k(S, keep_rate=0.99):
    n = len(S)
    for k in range(1, n + 1):
        reverse_rate = np.sum(S[:k]) / np.sum(S)
        if reverse_rate >= keep_rate:
            return k

#原始数据投影到主成分上
def project(X, U, k):
    Ureduce = U[:,:k]
    Z = X.dot(Ureduce)

    return Z


#恢复数据
def recover(Z, U, k):
    Ureduce = U[:,:k]
    X_recover = Z.dot(Ureduce.T)

    return X_recover

#整合流程
def keep_pca_information(X, keep_rate=0.99):
    U, S, V = pca(X)
    k = choose_k(S, keep_rate)
    Z = project(X, U, k)
    X_recover = recover(Z, U, k)

    return X_recover

'''人脸主成分保留'''
faces = loadmat('D:\Python\code/7-kmeans and PCA\data/faces.mat')
X = faces['X']

def plot_n_image(X, n):
    grid_size = int(np.sqrt(n))
    pic_size = int(np.sqrt(X.shape[1]))
    fig, ax = plt.subplots(grid_size, grid_size)
    i = 0
    for r in range(grid_size):
        for c in range(grid_size):
            ax[r,  c].imshow(np.reshape(X[i], (pic_size, pic_size)).T)
            i += 1
    plt.show()


X_recover = keep_pca_information(X, keep_rate=0.9)
plot_n_image(X_recover, 7)
