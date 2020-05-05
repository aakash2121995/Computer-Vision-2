import numpy as np
from numpy.linalg import inv

##############################################################################################################
#Auxiliary functions for Regression
##############################################################################################################
#returns features with bias X (num_samples*(1+num_features)) and target values Y (num_samples*target_dims)
def read_data_reg(filename):
    data = np.loadtxt(filename)
    Y = data[:,:2]
    X = np.concatenate((np.ones((data.shape[0], 1)), data[:,2:]), axis=1)
    return Y, X

#takes features with bias X (num_samples*(1+num_features)) and target values Y (num_samples*target_dims)
#returns regression coefficients w ((1+num_features)*target_dims)
def lin_reg(X, Y):
    return inv(X@X.T)@X@Y

#takes features with bias X (num_samples*(1+num_features)), target Y (num_samples*target_dims) and regression coefficients w ((1+num_features)*target_dims)
#returns fraction of mean square error and variance of target prediction separately for each target dimension
def test_lin_reg(X, Y, w):
    Y_pred = X@w
    return np.mean(np.square(Y - Y_pred),axis=0)/np.var(Y,axis=0)

#takes features with bias X (num_samples*(1+num_features)), centers of clusters C (num_clusters*(1+num_features)) and std of RBF sigma
#returns matrix with scalar product values of features and cluster centers in higher embedding space (num_samples*num_clusters)
def RBF_embed(X, C, sigma):
    embedding = np.zeros((X.shape[0],C.shape[0]))
    for center_id in range(C.shape[0]):
        center = C[center_id]
        embedding[:, center_id] = np.exp(-np.sum((X - center) ** 2, axis=1) / (2*sigma**2))
    return embedding
############################################################################################################
#Linear Regression
############################################################################################################

def run_lin_reg(X_tr, Y_tr, X_te, Y_te):

    print('MSE/Var linear regression')
    w = lin_reg(X_tr.T,Y_tr)
    err = test_lin_reg(X_te,Y_te,w)
    print(err)

############################################################################################################
#Dual Regression
############################################################################################################
def run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    # Parameters for Best Model
    best_error = 10 ** 5 * np.ones((2,))
    opt_sigma = 0

    # Train Validation Split
    X_train = X_tr[tr_list]
    Y_train = Y_tr[tr_list]
    Y_val = Y_tr[val_list]
    X_val = X_tr[val_list]

    for sigma_pow in range(-5, 3):
        sigma = np.power(3.0, sigma_pow)

        # Kernel Embeddings and regression
        X_T_X_ = RBF_embed(X_train,X_train,sigma)
        w = lin_reg(X_T_X_, Y_train)
        X_T_X_val = RBF_embed(X_val,X_train, sigma)
        err_dual = test_lin_reg(X_T_X_val, Y_val, w)

        # Validation Error
        print('MSE/Var dual regression for val sigma='+str(sigma))
        print(err_dual)
        if all(err_dual - best_error < 0):
            opt_sigma = sigma
            best_error = err_dual

    # Recomputation with the best sigma
    X_T_X_ = RBF_embed(X_train, X_train, opt_sigma)
    w = lin_reg(X_T_X_, Y_train)

    # Test Error Computation
    X_T_X_test = RBF_embed(X_te, X_train, opt_sigma)
    err_dual = test_lin_reg(X_T_X_test, Y_te, w)
    print('MSE/Var dual regression for test sigma='+str(opt_sigma))
    print(err_dual)

############################################################################################################
#Non Linear Regression
############################################################################################################
def run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    from sklearn.cluster import KMeans
    best_error = 10**5*np.ones((2,))
    opt_sigma, opt_num_clusters = 0, 0
    X_train = X_tr[tr_list]
    Y_train = Y_tr[tr_list]
    Y_val = Y_tr[val_list]
    X_val = X_tr[val_list]
    for num_clusters in [10, 30, 100]:
        # Running Kmeans
        kmeans = KMeans(num_clusters)
        kmeans.fit(X_train)
        clusters = kmeans.cluster_centers_
        for sigma_pow in range(-5, 3):
            sigma = np.power(3.0, sigma_pow)

            # RBF Kernel application for Regression
            Z_train = RBF_embed(X_train,clusters,sigma)
            w = lin_reg(Z_train.T, Y_train)
            Z_val  = RBF_embed(X_val,clusters,sigma)
            err_dual = test_lin_reg(Z_val, Y_val, w)

            # Saving Optimal Parameters
            if  all(err_dual - best_error < 0):
                best_error = err_dual
                opt_sigma = sigma
                opt_num_clusters = num_clusters
            print('MSE/Var non linear regression for val sigma='+str(sigma)+' val num_clusters='+str(num_clusters))
            print(err_dual)

    # Recomputation with best parameters
    print('MSE/Var non linear regression for test sigma='+str(opt_sigma)+' test num_clusters='+str(opt_num_clusters))
    kmeans = KMeans(opt_num_clusters)
    kmeans.fit(X_train)
    clusters = kmeans.cluster_centers_
    Z_train = RBF_embed(X_train, clusters, opt_sigma)
    w = lin_reg(Z_train.T, Y_train)
    Z_test = RBF_embed(X_te, clusters, opt_sigma)
    err_dual = test_lin_reg(Z_test, Y_te, w)
    print(err_dual)

####################################################################################################################################
#Auxiliary functions for classification
####################################################################################################################################
#returns features with bias X (num_samples*(1+num_feat)) and gt Y (num_samples)
def read_data_cls(split):
    feat = {}
    gt = {}
    for category in [('bottle', 1), ('horse', -1)]: 
        feat[category[0]] = np.loadtxt('data/'+category[0]+'_'+split+'.txt')
        feat[category[0]] = np.concatenate((np.ones((feat[category[0]].shape[0], 1)), feat[category[0]]), axis=1)
        gt[category[0]] = category[1] * np.ones(feat[category[0]].shape[0])
    X = np.concatenate((feat['bottle'], feat['horse']), axis=0)
    Y = np.concatenate((gt['bottle'], gt['horse']), axis=0)
    return Y, X

#takes features with bias X (num_samples*(1+num_features)), and current_parameters w (num_features+1)
#Returns the predicted value of label
epsilon = 1e-4
def predict(X, w):
    Z = np.matmul(X, w)
    Y_output = 1.0 / ((1.0 + np.exp(-Z ))+ epsilon)
    return Y_output

# takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
# returns gradient with respect to w (num_features)
def log_llkhd_grad(X, Y, w):
    Y_output = predict(X,w)
    grad_weight = np.matmul(X.T, Y_output-(Y+1)/2)

    return grad_weight

#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns log likelihood loss
def get_loss(X, Y, w):
    Y_output  = predict(X, w)
    loss = -(np.dot((Y.T+1)/2, np.log(Y_output)) + np.dot(1 - (Y.T+1)/2, np.log(1 - Y_output)))
    return loss

#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns accuracy
def get_accuracy(X, Y, w):
    Y_output = predict(X, w)
    predicted = []
    for i in range(len(Y)):
        if Y_output[i] > 0.5:
            predicted.append(1)
        else:
            predicted.append(-1)
    return np.count_nonzero(predicted == Y) / len(Y)

####################################################################################################################################
#Classification
####################################################################################################################################
def run_classification(X_tr, Y_tr, X_te, Y_te, step_size):
    print('classification with step size '+str(step_size))
    max_iter = 10000
    W = np.random.uniform(0, 1, size=(X_tr.shape[1]))
    for step in range(max_iter):

        grad_weight = log_llkhd_grad(X_tr, Y_tr, W)
        W = W - step_size * grad_weight
        if step%1000 == 0:
            loss = get_loss(X_tr, Y_tr, W)
            accuracy = get_accuracy(X_tr, Y_tr, W)
            print('step='+str(step)+' loss='+str(loss)+' accuracy='+str(accuracy))
    test_set_loss = get_loss(X_te, Y_te, W)
    accuracy = get_accuracy(X_te, Y_te, W)
    print('test set loss='+str(test_set_loss)+' accuracy='+str(accuracy))


####################################################################################################################################
#Exercises
####################################################################################################################################
Y_tr, X_tr = read_data_reg('data/regression_train.txt')
Y_te, X_te = read_data_reg('data/regression_test.txt')

run_lin_reg(X_tr, Y_tr, X_te, Y_te)

tr_list = list(range(0, int(X_tr.shape[0]/2)))
val_list = list(range(int(X_tr.shape[0]/2), X_tr.shape[0]))

run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)
run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)

step_size = 0.1
Y_tr, X_tr = read_data_cls('train')
Y_te, X_te = read_data_cls('test')
run_classification(X_tr, Y_tr, X_te, Y_te, step_size)

#Solution 2.1(Theory)
# The model got stuck in local minima beacause of low learning rate, as a too low learning rate will either take too
# long to converge or get stuck in an undesirable local minimum.