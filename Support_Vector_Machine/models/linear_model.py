"""Linear model base class."""

import abc
import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class LinearModel(object):
    """Abstract class for linear models."""

    def __init__(self, ndims, w_init='zeros', w_decay_factor=0.001):
        """Initialize a linear model.

        This function prepares an uninitialized linear model.
        It will initialize the weight vector, self.w, based on the method
        specified in w_init.

        We assume that the last index of w is the bias term, self.w = [w,b]

        self.w(numpy.ndarray): array of dimension (n_dims+1,1)

        w_init needs to support:
          'zeros': initialize self.w with all zeros.
          'ones': initialze self.w with all ones.
          'uniform': initialize self.w with uniform random number between [0,1)

        Args:
            ndims(int): feature dimension
            w_init(str): types of initialization.
            w_decay_factor(float): Weight decay factor.
        """
        self.ndims = ndims
        self.w_init = w_init
        self.w_decay_factor = w_decay_factor
        self.w = None
        self.x = None
        # Implementation here.
        if w_init == 'zeros':
            self.w = np.zeros((self.ndims + 1,1))  #shape (17,)
            #pass
        elif w_init == 'ones':
            #pass
            self.w = np.ones ((self.ndims + 1,1))
        elif w_init == 'uniform':
            self.w = np.random.random ((self.ndims + 1,1))
            #pass

    def forward(self, x):
        """Forward operation for linear models.

        Performs the forward operation. Appends 1 to x then compute
        f=w^Tx, and return f.

        Args:
            x(numpy.ndarray): Dimension of (N, ndims), N is the number
              of examples.

        Returns:
            (numpy.ndarray): Dimension of (N,1)
        """
        N = np.shape(x)[0]
        #print("initial x shape:", np.shape(x))
        f = []
        temp_x = []
        for i in range(N):
            temp = x[i].tolist()
            temp.append(1)
            temp_x.append(temp)
        x_matrix = np.array(temp_x)
        self.x = x_matrix # (N,193)  UNSURE self.x needed for backward
        f = np.matmul(x_matrix, self.w)
    
        # Implementation here.
        #pass

        return f

    @abc.abstractmethod
    def backward(self, f, y):
        """Do not need to be implemented here."""
        pass

    @abc.abstractmethod
    def total_loss(self, f, y):
        """Do not need to be implemented here."""
        pass

    def predict(self, f):
        """Do not need to be implemented here."""
        pass
