import numpy as np

#代价和梯度
def cost_and_grad(params, num_features, Y, R, lambd):
    num_movie , num_user = Y.shape
    X = np.reshape(params[:num_movie * num_features], (num_movie, num_features))
    theta = np.reshape(params[num_movie * num_features:], (num_user, num_features))

    error = np.multiply(X.dot(theta.T) - Y, R)
    
    square_error = 1/2 * np.sum(np.square(error))
    reg_error = lambd/2 * (np.sum(np.square(X)) + np.sum(np.square(theta)))
    cost = square_error  + reg_error

    X_grad = error.dot(theta) + lambd * X
    theta_grad = error.T.dot(X) + lambd * theta

    grad = np.concatenate((X_grad.ravel(), theta_grad.ravel()))

    return cost , grad

#加载评分数据
from scipy.io import loadmat
data = loadmat('D:\Python\code\8-anomaly detection and recommendation\data\movies_rating.mat')
Y , R = data['Y'] , data['R']

#加载其他相关参数：电影数量、电影特征数、用户数
params_data = loadmat('D:\Python\code\8-anomaly detection and recommendation\data\movieParams.mat')
num_users, num_movies, num_features = int(params_data['num_users']), int(params_data['num_movies']), int(params_data['num_features'])

#新用户评分
my_rating = np.zeros((num_movies, 1))
my_rating[0] = 4
my_rating[6] = 3
my_rating[11] = 5
my_rating[53] = 4
my_rating[63] = 5
my_rating[65] = 3
my_rating[68] = 5
my_rating[97] = 2
my_rating[182] = 4
my_rating[225] = 5
my_rating[354] = 5

#新用户评分加入到所有用户数据
Y = np.c_[Y, my_rating]
R = np.c_[R, np.where(my_rating > 0, 1, 0)]

#评分减去均值
Ynorm = np.zeros((Y.shape))
Ymean = np.zeros((num_movies, 1))
for i in range(num_movies):
    idx = np.where(R[i] == 1)[0]
    Ymean[i] = Y[i, idx].mean()
    Ynorm[i, idx] = Y[i, idx] - Ymean[i]

#随机初始化电影特征参数和用户参数
X = np.random.randn(num_movies, num_features)
theta = np.random.randn(num_users+1, num_features)
params  = np.concatenate((X.ravel(), theta.ravel()))


#优化
from scipy.optimize import minimize
lambd = 10
res = minimize(fun=cost_and_grad, x0=params, args=(num_features, Ynorm, R, lambd), method='CG',  jac=True)
print(res)

params = res.x

X = np.reshape(params[:num_movies * num_features], (num_movies, num_features))
theta = np.reshape(params[num_movies * num_features:], (num_users+1, num_features))

#预测评分
predictions = X.dot(theta.T)

my_pred = predictions[:,-1].reshape(-1,1) + Ymean

ranking = np.argsort(my_pred, axis=0)[::-1]

#加载电影数据
movies = {}
f = open('D:\Python\code\8-anomaly detection and recommendation\data\movie_ids.txt')
for line in f:
    name = line.split(' ')
    movies[int(name[0]) - 1] = ' '.join(name[1:])

#推荐电影
print('The 10 movies recommend for you:')
for i in range(10):
    print(str(i+1)+'、'+ movies[int(ranking[i])] + ', predict rating:'+ str(float(my_pred[int(ranking[i])])))