"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression
from numpy.linalg import inv


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.
eval_model
    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """

    x = processed_dataset[0]
    N = np.shape(x)[0]
    y = processed_dataset[1]
    epoch = 0
    count = 0
    while count is not (num_steps-1):
        x_split = np.array_split(x, int(1000 / batch_size))
        y_split = np.array_split(y, int(1000 / batch_size))
        num_batch = np.shape(x_split)[0]
        for i in range(num_batch):
            if(count == num_steps - 1):
                print(model.w)
                return model
            else:
                model.x = x_split[i]
                model.y = y_split[i]
                update_step(model.x, model.y, model, learning_rate)
                count += 1
        epoch += 1
        x_y = np.concatenate((x, y), axis=1)
        np.random.shuffle(x_y)
        temp_y = []
        temp_ndims = np.shape(x_y)[1]
        for i in range(N):
            temp_y.append([x_y[i][temp_ndims-1]])
        y = np.array(temp_y)
        n = temp_ndims - 1
        x = np.delete(x_y, np.s_[n:], axis=1)
    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """

    f = model.forward(x_batch)
    gd = model.backward(f, y_batch)
    model.w = model.w-learning_rate * gd
    return 0


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    x = processed_dataset[0]
    y = processed_dataset[1]
    N = np.shape(x)[0]
    temp_x = []
    for i in range(N):
        temp = x[i].tolist()
        temp.append(1)
        temp_x.append(temp)
    x_matrix = np.array(temp_x)
    id_matrix = np.identity(model.ndims + 1)
    m = model.w_decay_factor * id_matrix
    mul1 = inv(np.add(np.matmul(np.transpose(x_matrix), x_matrix), m))
    mul2 = np.matmul(mul1, np.transpose(x_matrix))
    cf = np.matmul(mul2, y)
    model.w = cf
    return cf


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    x = processed_dataset[0]
    y = processed_dataset[1]
    loss_sum = 0
    N = np.shape(x)[0]
    model.forward(x)
    ndims = np.shape(model.x)[1]
    mul = np.matmul(model.x, model.w)
    loss_vec = mul - y
    for i in range(N):
        loss_sum += loss_vec[i][0] * loss_vec[i][0]
    loss_sum = loss_sum / 2

    w_sum = 0
    for j in range(ndims):
        w_sum += (model.w[j][0])*(model.w[j][0])
    w_sum = model.w_decay_factor / 2 * w_sum
    loss = loss_sum+w_sum
    return loss
