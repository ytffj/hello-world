import numpy as np
import math

'''随机小批量'''
def random_minibatches(X, Y, minibatch_size=64, seed=0):
    np.random.seed(seed)

    m   = X.shape[1]
    shuffle_idx = np.random.permutation(X.shape[1])
    shuffle_X = X[:, shuffle_idx]
    shuffle_Y = Y[:, shuffle_idx]
    num_minibatch = math.ceil(m / minibatch_size)
    minibatches = []
    for k in range(num_minibatch):
        minibatch_X = shuffle_X[:, k * minibatch_size:(k + 1) * minibatch_size]
        minibatch_Y = shuffle_Y[:, k * minibatch_size:(k + 1) * minibatch_size]

        minibatches.append((minibatch_X, minibatch_Y))
    
    return minibatches

def update_params(params, grads, learning_rate):
    L = len(params) // 2
    for l in range(1, L + 1):
        params['W'+str(l)] -= learning_rate * grads['dW'+str(l)]
        params['b'+str(l)] -= learning_rate * grads['db'+str(l)]

    return params

'Momentum'
def initial_momentum(params):
    L = len(params) // 2
    V = {}

    for l in range(L):
        V['dW'+ str(l + 1)] =  np.zeros((params['W'+ str(l + 1)].shape))
        V['db'+ str(l + 1)] =  np.zeros((params['b'+ str(l + 1)].shape))
    
    return V

def update_params_with_momentum(params, grads, V, learning_rate, β):
    L = len(params) // 2
    for l in range(L):
        V['dW'+ str(l + 1)] = β * V['dW'+ str(l + 1)] + (1 - β) * grads['dW'+ str(l + 1)]
        V['db'+ str(l + 1)] = β * V['db'+ str(l + 1)] + (1 - β) * grads['db'+ str(l + 1)]

        params['W'+ str(l + 1)] = params['W'+ str(l + 1)] - learning_rate * V['dW'+ str(l + 1)]
        params['b'+ str(l + 1)] = params['b'+ str(l + 1)] - learning_rate * V['db'+ str(l + 1)]
    
    return params, V

'''Adam'''
def initial_adam(params):
    L = len(params) // 2
    v, s = {}, {}
    for l in range(L):
        v['dW'+ str(l + 1)] = np.zeros((params['W'+ str(l + 1)].shape))
        v['db'+ str(l + 1)] = np.zeros((params['b'+ str(l + 1)].shape))

        s['dW'+ str(l + 1)] = np.zeros((params['W'+ str(l + 1)].shape))
        s['db'+ str(l + 1)] = np.zeros((params['b'+ str(l + 1)].shape))
    
    return v, s

def update_params_with_adam(params, grads, v, s, t, β1=0.9, β2=0.999, ε=1e-8, learning_rate=0.01):
    L = len(params) // 2
    for l in range(1, L + 1):
        v['dW'+ str(l)] = β1 * v['dW'+ str(l)] + (1 - β1) * grads['dW'+ str(l)]
        v['db'+ str(l)] = β1 * v['db'+ str(l)] + (1 - β1) * grads['db'+ str(l)]

        v_dw_correct  = v['dW'+ str(l)] / (1 - β1 ** t)
        v_db_correct  = v['db'+ str(l)] / (1 - β1 ** t)

        s['dW'+ str(l)] = β2 * s['dW'+ str(l)] + (1 - β2) * grads['dW'+ str(l)] ** 2
        s['db'+ str(l)] = β2 * s['db'+ str(l)] + (1 - β2) * grads['db'+ str(l)] ** 2

        s_dw_correct  = s['dW'+ str(l)] / (1 - β2 ** t)
        s_db_correct  = s['db'+ str(l)] / (1 - β2 ** t)

        params['W'+ str(l)] -= learning_rate  * v_dw_correct / (np.sqrt(s_dw_correct) + ε)
        params['b'+ str(l)] -= learning_rate  * v_db_correct / (np.sqrt(s_db_correct) + ε)

    return params, v, s

'''梯度优化器'''
import opt_utils
import matplotlib.pyplot as plt
def optimizer(X, Y, layer_size,optimize_type, minibatch_size=64, β1=0.9, β2=0.999, ε=1e-8, learning_rate=0.008, iterations=10000, print_cost=False, plot=False):
    params = opt_utils.initialize_parameters(layer_size)
    seed = 0
    num_minibatch = math.ceil(Y.shape[0] / minibatch_size)
    if optimize_type == 'momentum':
        V = initial_momentum(params)
    elif optimize_type == 'adam':
        v, s = initial_adam(params)

    costs = []
    for i in range(iterations):
        minibatches = random_minibatches(X, Y, minibatch_size, seed)
        seed += 1
        for t in range(num_minibatch):
            minibatch_X , minibatch_Y = minibatches[t]
            
            A3, cache = opt_utils.forward_propagation(minibatch_X, params)
            grads = opt_utils.backward_propagation(minibatch_X, minibatch_Y, cache)

            if optimize_type == 'momentum':
                params, V = update_params_with_momentum(params, grads, V, learning_rate, β1)
            elif optimize_type == 'adam':
                params, v, s = update_params_with_adam(params, grads, v, s, t+1, β1, β2, ε, learning_rate)
            else:
                params = update_params(params, grads, learning_rate)
        
        if i %  100 == 0:
            cost = opt_utils.compute_cost(A3, minibatch_Y)
            costs.append(cost)
            if print_cost and i % 1000 == 0:
                print('第',i,'次迭代代价为:',cost)
    if plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations(per 100)')
        plt.title('learning rate = '+ str(learning_rate) +' with '+ optimize_type)
        plt.show()
    
    return params

train_X, train_Y= opt_utils.load_dataset()
layers_dims = [train_X.shape[0],5,2,train_Y.shape[0]]
optimize_type = 'adam'
params = optimizer(train_X, train_Y, layers_dims, optimize_type, print_cost=True, plot=True)

prediction = opt_utils.predict(train_X, train_Y, params)
plt.title(optimize_type + ' optimizer')
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(params, x.T), train_X, train_Y)