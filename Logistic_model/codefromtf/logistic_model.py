"""logistic model class for binary classification."""
import tensorflow as tf
import numpy as np

class LogisticModel_TF(object):

    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of Weight is the bias term,
            Weight = [Bias, W1, W2, W3, ...]
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
        self.W0 = None
        self.W = None
        self.x = None
        self.y = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            self.W0 = np.zeros([self.ndims+1,1])
            # Hint: self.W0 = tf.zeros([self.ndims+1,1])
            #pass
        elif W_init == 'ones':
            self.W0 = np.ones([self.ndims+1,1])
            #pass
        elif W_init == 'uniform':
            self.W0 = np.transpose([np.random.random (self.ndims + 1)])
            #pass
        elif W_init == 'gaussian':
            self.W0 = np.transpose([np.random.normal (0 , 0.1, self.ndims + 1)])
        else:
            print ('Unknown W_init ', W_init)


    def build_graph(self, learn_rate):
        """ build tensorflow training graph for logistic model.
        Args:
            learn_rate: learn rate for gradient descent
            ......: append as many arguments as you want
        """
        self.W = tf.Variable(self.W0)
        self.x = tf.placeholder(tf.float64, shape = (10,self.ndims + 1))
        self.y = tf.placeholder(tf.float64, shape = (10,1))
        sig = tf.sigmoid(tf.matmul(self.x,self.W))
        loss = tf.losses.mean_squared_error(self.y, sig)
        train = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
        return train
        ###############################################################
        # Fill your code in this function
        ###############################################################
        # Hint: self.W = tf.Variable(self.W0)
       

    def fit(self, Y_true, X, max_iters):
        """ train model with input dataset using gradient descent.
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,1)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            max_iters: maximal number of training iterations
            ......: append as many arguments as you want
        Returns:
            (numpy.ndarray): sigmoid output from well trained logistic model, used for classification
                             with a dimension of (# of samples, 1)
        """
        if(self.W_init == "zeros"):
           max_iters = 100000
           learn_rate = 0.0001
        if(self.W_init == "ones"):
           max_iters = 100000
           learn_rate = 0.01 
        if(self.W_init == "uniform"):
           max_iters = 100000
           learn_rate = 0.001
        if(self.W_init == "gaussian"):
           max_iters = 1000
           learn_rate = 0.01
        train = self.build_graph(learn_rate)
        model = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(model)
        for epoch in range(max_iters):
            idx = np.random.choice(np.shape(X)[0], 10, replace = False)
            a=  sess.run(train, feed_dict = {self.x: X[idx], self.y: Y_true[idx]})
        w_arr = sess.run(self.W)
        N = np.shape(X)[0]
        score_arr = []
        predict = []
        for i in range(N):
            wTx = np.dot(np.transpose(w_arr), X[i])
            score = float(1/(1+np.exp(-wTx)))
            score_arr.append (score)
            if(score_arr[i] >= 0.5):
                predict.append(1)
            else: 
                predict.append(0)
            print("score:",score, "label",Y_true[i])
        accuracy = 0
        for k in range(N):
            if (predict[k] == Y_true[k]):
                accuracy += 1
        score_arr = np.transpose([score_arr])
        #print(score_arr)
        print("shape:",np.shape(score_arr))
        print("accuracy:",accuracy)
        return score_arr
        #tf.gradients(loss)
        ###############################################################
        # Fill your code in this function
        ###############################################################
        #$pass
