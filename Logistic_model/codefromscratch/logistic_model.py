# -*- coding: utf-8 -*-
"""logistic model class for binary classification."""

import numpy as np

class LogisticModel(object):

    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of W is the bias term,
            self.W = [Bias, W1, W2, W3, ...]
            where Wi correspnds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random number between [0,1)
          'gaussian': initialize self.W with gaussion distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims
        self.W_init = W_init
        self.W = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            self.W = np.zeros(self.ndims + 1)  #shape (17,)
            #pass
        elif W_init == 'ones':
            #pass
            self.W = np.ones (self.ndims + 1)
        elif W_init == 'uniform':
            self.W = np.random.random (self.ndims + 1)
            #pass
        elif W_init == 'gaussian':
            self.W = np.random.normal (0 , 0.1, self.ndims + 1)
            #pass
        else:
            print ('Unknown W_init ', W_init)

    def save_model(self, weight_file):
        """ Save well-trained weight into a binary file.
        Args:
            weight_file(str): binary file to save into.
        """
        self.W.astype('float32').tofile(weight_file)
        print ('model saved to', weight_file)

    def load_model(self, weight_file):
        """ Load pretrained weghit from a binary file.
        Args:
            weight_file(str): binary file to load from.
        """
        self.W = np.fromfile(weight_file, dtype=np.float32)
        print ('model loaded from', weight_file)

    def forward(self, X):
        """ Forward operation for logistic models.
            Performs the forward operation, and return probability score (sigmoid).
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): probability score of (label == +1) for each sample
                             with a dimension of (# of samples,)
        """
        N = np.shape(X)[0]
        score_arr = []
        for i in range(N):
            wTx = np.dot(np.transpose(self.W), X[i])
            #print ("wTx:", wTx)
            score = 1/(1+np.exp(-wTx))
            score_arr.append (score)
        return score_arr
        ###############################################################
        # Fill your code in this function
        ###############################################################
        #pass

    def backward(self, Y_true, X):
        """ Backward operation for logistic models.
            Compute gradient according to the probability loss on lecture slides
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
        Returns:
            (numpy.ndarray): gradients of self.W
        """

        gradient_sum = np.zeros(self.ndims+1)
        N = np.shape(Y_true)[0]
        for i in range(N):
            y_x = Y_true[i] * X[i]
            #print([X[i]], "w:::", self.W, np.shape(self.W), np.shape(X[i]))
            wTx = np.dot(np.transpose(self.W),X[i])
            gradient = -y_x * np.exp(-Y_true[i] * wTx)/(1+np.exp(-Y_true[i] * wTx))
            gradient_sum = gradient_sum + gradient
        return gradient_sum


        ###############################################################
        # Fill your code in this function
        ###############################################################
        #pass

    def classify(self, X):
        """ Performs binary classification on input dataset.
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): predicted label = +1/-1 for each sample
                             with a dimension of (# of samples,)
        """
        scores = self.forward(X)
        predict = []
        for score in scores:
            if(score >= 0.5):
                predict.append(1)
                print("score+1:", score)
            else:
                predict.append(-1)
                print("score-1:",score)
        return predict

        ###############################################################
        # Fill your code in this function
        ###############################################################

        #pass

    def fit(self, Y_true, X, learn_rate, max_iters):
        """ train model with input dataset using gradient descent.
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            learn_rate: learning rate for gradient descent
            max_iters: maximal number of iterations
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################

        N = np.shape(Y_true)[0]
        count = 0
        while count is not (max_iters - 1):
            x_split = np.array_split(X, int(N / 10 ))
            y_split = np.array_split(Y_true, int(N / 10))
            num_batch = np.shape(x_split)[0]
            for i in range(num_batch):
                if(count == max_iters - 1):
                    #print(self.W)
                    #print("WWW", self.W)
                    return
                else:
                    batch_x = x_split[i]
                    batch_y = y_split[i]
                    self.W = self.W- learn_rate* self.backward(Y_true, X)
                    #print(count)
                    count += 1
            #print("Y transpose:",np.shape(np.transpose([Y_true])))
            x_y = np.concatenate((X, np.transpose([Y_true])), axis=1)
            #print("concatenate;::", x_y[0])
            np.random.shuffle(x_y)
            temp_y = []
            temp_ndims = np.shape(x_y)[1]
            for i in range(N):
                temp_y.append([x_y[i][temp_ndims-1]])
            #after append, np.array, dimension = (N,)
            Y_true = np.array(temp_y)
            Y_true = Y_true.reshape((N,))
            #print("yshape",np.shape(Y_true))
            n = temp_ndims - 1
            #print(y)
            X = np.delete(x_y, np.s_[n:], axis=1)
            #print("xshape",np.shape(X))
            #print(x[0])
        return
            #pass
