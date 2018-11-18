"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """
    x = data['image']
    y = data['label']
    N = np.shape(x)[0]
    epoch = 0
    count = 0
    while count is not (num_steps-1):
        x_split = np.array_split(x,int(N/batch_size))
        y_split = np.array_split(y,int(N/batch_size))
        #print("num of batch:",np.shape(x_split))
        num_batch = np.shape(x_split)[0]
        for i in range(num_batch):
            if(count == num_steps - 1):
                print("final w:",model.w)
                #print("final count:", count)
                return model
            else:
                #model.x = x_batch[i] #UNSURE  #without append 1
                #model.y = y_batch[i]  #UNSURE  #without append 1
                x_batch = x_split[i]
                #print("size of third split:", np.shape(x_split[3]))
                y_batch = y_split[i]
                update_step(x_batch, y_batch, model, learning_rate) #UNSURE
                count += 1
                #print(count)
                #print("updated w:",np.shape(model.w))
                #return model
            #break
        epoch += 1
        # x: (N,193)  y:(N,1)
        # shuffle whole data for the next epoch
        x_y = np.concatenate((x,y), axis = 1)
        np.random.shuffle(x_y)
        temp_y = []
        temp_ndims = np.shape(x_y)[1]
        for i in range(N):
            temp_y.append([x_y[i][temp_ndims-1]])
        y  = np.array(temp_y)
        n = temp_ndims - 1
        x = np.delete(x_y, np.s_[n:], axis=1)
    return model


    # Performs gradient descent. (This function will not be graded.)
    #pass


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    f = model.forward(x_batch)
    model.w = model.w - learning_rate*model.backward(f,y_batch)
    # Implementation here. (This function will not be graded.)
    #pass


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    print("P",P.shape,"q",q.shape,"G",G.shape, "h",h.shape)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)
    #pass
    # Set model.w
    print("shape of z:", np.shape(z) )
    w = []
    for i in range(model.ndims+1):
        temp_w = z[i][0]
        w.append(temp_w)
    model.w = np.array(w)
    #print("model.w:",model.w)
    print("model.w shape:",np.shape(model.w))

def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    x = data['image']
    N = np.shape(x)[0]
    #print("initial x shape:", np.shape(x))
    temp_x = []
    #print("original 1st in x_matrix:", x[0])
    #print("oriinal 2nd in x_matrix:", x[1])
    for i in range(N):
        temp = x[i].tolist()
        temp.append(1)
        temp_x.append(temp)
    x = np.array(temp_x)
    P = None
    q = None
    G = np.zeros((N+N,model.ndims+1+N))
    h = None
    P = np.identity(model.ndims+1+N)
    for i in range(model.ndims+1,model.ndims+1+N):
        P[i][i] = 0
    q_0 = np.full((model.ndims+1,1), 0)
    q_1 = np.full((N,1),1)
    q = np.concatenate((q_0,q_1), axis = 0)
    #print("q shape:", np.shape(q))
    for i in range(N):
        for j in range(model.ndims+1):
            G[i][j] = - data['label'][i][0] * x[i][j]
            G[i][i+model.ndims+1] = -1
    for i in range(N,2*N):
            G[i][i+1+model.ndims-N]  = -1
    h_1 = np.full((N,1), -1)
    h_0 = np.full((N,1),0)
    h = np.concatenate((h_1,h_0), axis = 0)
    print("shape of G",np.shape(G))
    print("shape of h:", np.shape(h))
    print("shape of q:",np.shape(q))
    print("shape of p:",np.shape(P))

    # Implementation here.
    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.

    #print("initial x shape:", np.shape(x))
    N = np.shape(data['image'])[0]
    f = []
    temp_x = []
    loss = 0
    acc = 0
    f = model.forward(data['image'])
    y_predict  = model.predict(f)
    for i in range(N):
        #print(f[i], y_predict[i],data['label'][i])
        if(y_predict[i] == data['label'][i]):
            acc+= 1
    acc = float(acc/N)
    loss = model.total_loss(f,data['label'])

    return loss, acc
