"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation based on the loss in total_loss.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1, 1).
        """
        reg_grad = None
        loss_grad = None
        # Implementation here.
        #pass
        N = np.shape(self.x)[0]
        #print("N:",N)
        grad_sum = np.zeros((1,self.ndims+1))
        for i in range(N):
            wTx = np.matmul([self.x[i]], self.w)  #193
            hinge = y[i] * wTx
            if(hinge< 1):
                grad_sum = grad_sum -y[i]* self.x[i]
        reg_grad = np.transpose(grad_sum)
        loss_grad = self.w_decay_factor * self.w
        total_grad = reg_grad + loss_grad
        #print("type of total_grad:", type(total_grad))
        #print("shape of total_grad:", np.shape(total_grad))
        return total_grad

    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.
        Total loss is hinge_loss + w_decay_factor/2*||w||^2

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """
        N = np.shape(self.x)[0]
        #self.forward(self.x)  # UNSURE
        hinge_loss = 0
        l2_loss = 0
        for i in range(N):
            wTx = np.matmul([self.x[i]], self.w)  #193
            #print(wTx)
            wTx = np.reshape(wTx, (1,))
            hinge = max(1 - y[i][0] * wTx[0], 0)
            hinge_loss = hinge_loss + hinge
        #print("type of wTx:", wTx)
        #print("hinge:",type(hinge))
        #print("type of hinge:",hinge)
        # Implementation here.
        #pass
        iden = np.identity(self.ndims+1)
        zTi = np.matmul(np.transpose(self.w), iden)
        l2_loss = self.w_decay_factor/2 * np.matmul(zTi,self.w)  #UNSURE
        #print("l2_loss type:", l2_loss, "value",l2_loss)
        l2_loss = l2_loss.item()
        #print("hinge_loss type:", hinge_loss,"value",hinge_loss)
        total_loss = hinge_loss + l2_loss
        print("type of total loss:",type(total_loss))
        return total_loss

    def predict(self, f):
        """Converts score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,1). Tie break 0 to 1.0.
        """
        y_predict = []
        # Implementation here.
        #pass
        for score in f:
            if(score >= 0):
                y_predict.append(1)
                #print("score+1:", score)
            else:
                y_predict.append(-1)
                #print("score-1:",score)
        y_predict = np.array(y_predict)
        print("type of predict:", type(y_predict))
        print("shape of predict:",np.shape(y_predict))
        return y_predict
