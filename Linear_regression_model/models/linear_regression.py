"""Implements linear regression."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LinearRegression(LinearModel):
    """Implements a linear regression mode model."""

    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        with respect to w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).

        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,1).
        """

        N = np.shape(y)
        diff = f - y
        total_grad = np.matmul(np.transpose(self.x), diff)
        return total_grad

    def total_loss(self, f, y):
        """Computes the total loss, square loss + L2 regularization.

        Overall loss is sum of squared_loss + w_decay_factor*l2_loss
        Note: Don't forget the 0.5 in the squared_loss!

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum square loss + reguarlization.
        """

        loss_sum = 0
        N = np.shape(self.x)[0]
        ndims = np.shape(self.x)[1]
        mul = np.matmul(self.x, self.w)
        loss_vec = mul-self.y
        for i in range(N):
            loss_sum += loss_vec[i][0] * loss_vec[i][0]
        loss_sum = loss_sum / 2

        w_sum = 0
        for j in range(ndims):
            w_sum += (self.w[j][0])*(self.w[j][0])
        w_sum = 0.0001/2 * w_sum
        tl = loss_sum+w_sum
        return tl

    def predict(self, f):
        """Nothing to do here.
        """
        return f
