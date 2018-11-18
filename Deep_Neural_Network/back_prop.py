import numpy as np

# define the number of iterations.
#num_itr = 1000
num_itr = 5
# define batch size.
batchSize = 3

# define the input data dimension.
inputSize = 2

# define the output dimension.
outputSize = 1

# define the dimension of the hidden layer.
hiddenSize = 3


class Neural_Network():
    def __init__(self):
        #weights
        #np.random.seed(101)
        self.U = np.random.randn(inputSize, hiddenSize)
        self.W = np.random.randn(hiddenSize, outputSize)
        self.e = np.random.randn(hiddenSize)
        self.f = np.random.randn(outputSize)


    def fully_connected(self, X, U, e):
        '''
        fully connected layer.
        inputs:
            U: weight
            e: bias
        outputs:
            X * U + e
        '''
        return np.dot(X, U) + e


    def sigmoid(self, s):
        '''
        sigmoid activation function.
        inputs: s
        outputs: sigmoid(s)
        '''
        #print("shpae of s", np.shape(s))
        #print("shape of sig:", np.shape(np.exp(-s)))
        return 1/(1+np.exp(-s))


    def sigmoidPrime(self, s):
        '''
        derivative of sigmoid (Written section, Part a).
        inputs:
            s = sigmoid(x)
        outputs:
            derivative sigmoid(x) as a function of s
        '''
        d_sigmoid = s - s*s
        return d_sigmoid


    def forward(self, X):
        '''
        forward propagation through the network.
        inputs:
            X: input data (batchSize, inputSize)
        outputs:
            c: output (batchSize, outputSize)
        '''
        layer1 = self.fully_connected(X, self.U, self.e)
        layer1_out = self.sigmoid(layer1)
        c = self.sigmoid(self.fully_connected(layer1_out,self.W, self.f))
        #c = []
        #X = layer1_out
        #W dimension: (hiddenSize, outputSize)
        '''
        for i in range(np.shape(X)[0]):  #for each sample
            x_i = np.reshape(X[i],(1,inputSize))
            #print("u origin:",self.U)
            #print("u next:",self.U[:,i])
            #print("x_i",np.shape(x_i))
            #print("uuuuu:",np.shape(self.U[:,i]))
            u_ij  = np.reshape(self.W[:,i],(inputSize,1))
            #print("shape of u_ij:", np.shape(u_ij))
            xu_sum = np.dot(X,u_ij)[0]
            #print("xu_sum shape:",type(xu_sum[0]))
            c.append(xu_sum[0])
        c = np.array(c)
        #print("ccc:",np.shape(c))
        e = np.reshape(self.e,(batchSize,outputSize))
        #print("e shape:",np.shape(e))
        #print("eee:",np.shape(self.e))
        c = c+e
        #print("c shape:",np.shape(c))
        '''
        #print("shape of layer1 out",np.shape(layer1_out))
        #print("shpae of e:",np.shape(self.e))
        return c


    def d_loss_o(self, gt, o):
        '''
        computes the derivative of the L2 loss with respect to
        the network's output.
        inputs:
            gt: ground-truth (batchSize, outputSize)
            o: network output (batchSize, outputSize)
        outputs:
            d_o: derivative of the L2 loss with respect to the network's
            output o. (batchSize, outputSize)
        '''

        d_o = 1/batchSize*(o - gt)
        return d_o


    def error_at_layer2(self, d_o, o):
        '''
        computes the derivative of the loss with respect to layer2's output
        (Written section, Part b
        inputs:
            d_o: derivative of the loss with respect to the network output (batchSize, outputSize)
            o: the network output (batchSize, outputSize)
        returns
            delta_k: the derivative of the loss with respect to the output of the second
            fully connected layer (batchSize, outputSize).
        '''

        delta_k = d_o * self.sigmoidPrime(o)
        return delta_k


    def error_at_layer1(self, delta_k, W, b):
        '''
        computes the derivative of the loss with respect to layer1's output (Written section, Part e).
        inputs:
            delta_k: derivative of the loss with respect to the output of the second
            fully connected layer (batchSize, outputSize).
            W: the weights of the second fully connected layer (hiddenSize, outputSize).
            b: the input to the second fully connected layer (batchSize, hiddenSize).
        returns:
            delta_j: the derivative of the loss with respect to the output of the second
            fully connected layer (batchSize, hiddenSize).
        '''

        delta_j = np.dot(np.dot(delta_k,W.T),self.sigmoidPrime(b))
        return delta_j




    # o = ck in hw  the network output
    #gt = tk in hw the ground truth


        #hidden, batch: 3;  input =  2   output = 1
        #delta_j = (E/hk) * wjk * g'(zj)
        #E/hk = error_at_layer2   (batchSize, outputSize)
        #b = g(zj)    (batchsize, hiddensize)    g'(zj) ()
        #W = (hiddenSize, outputSize)
        #(batchsize, output)(outputsize,hidden)(batchsize,hiddensize)

            #E/hk
            # o = ck in hw network output
            #fully_connected(X, self.U, self.e)
            #d_o dimension: (batchSize, outputSize)
            #o dimension: (batchSize, outputSize)


    def derivative_of_w(self, b, delta_k):
        '''
        computes the derivative of the loss with respect to W (Written section, Part c).
        inputs:
            b: the input to the second fully connected layer (batchSize, hiddenSize).
            delta_k: the derivative of the loss with respect to the output of the second
            fully connected layer's output (batchSize, outputSize).
        returns:
            d_w: the derivative of loss with respect to W  (hiddenSize ,outputSize).
        '''

        d_w = np.dot(b.T,delta_k)
        return d_w


    def derivative_of_u(self, X, delta_j):
        '''
        computes the derivative of the loss with respect to U (Written section, Part f).
        inputs:
            X: the input to the network (batchSize, inputSize).
            delta_j: the derivative of the loss with respect to the output of the first
            fully connected layer's output (batchSize, hiddenSize).
        returns:
            d_u: the derivative of loss with respect to U (inputSize, hiddenSize).
        '''
        d_u = np.dot(X.T,delta_j)
        return d_u


    def derivative_of_e(self, delta_j):
        '''
        computes the derivative of the loss with respect to e (Written section, Part g).
        inputs:
            delta_j: the derivative of the loss with respect to the output of the first
            fully connected layer's output (batchSize, hiddenSize).
        returns:
            d_e: the derivative of loss with respect to e (hiddenSize).
        '''
        d_e = np.sum(delta_j,axis = 0)
        return d_e


    #delta_k dimension: (batchSize, outputSize)
    #b dimension:(batchsize, hiddensize)
    #UNSURE

    def derivative_of_f(self, delta_k):
        '''
        computes the derivative of the loss with respect to f (Written section, Part d).
        inputs:
            delta_k: the derivative of the loss with respect to the output of the second
            fully connected layer's output (batchSize, outputSize).
        returns:
            d_f: the derivative of loss with respect to f (outputSize).
        '''
        d_f = np.sum(delta_k,axis =0)
        return d_f


    def backward(self, X, gt, o):
        '''
        backpropagation through the network.
        Task: perform the 8 steps required below.
        inputs:
            X: input data (batchSize, inputSize)
            y: ground truth (batchSize, outputSize)
            o: network output (batchSize, outputSize)
        '''
        #compute gradient for W, U, e and f
        # 1. Compute the derivative of the loss with respect to c.
        # Call: d_loss_o  return d_o
        d_o_c = self.d_loss_o(gt,o)


        # 2. Compute the error at the second layer (Written section, Part b).
        # Call: error_at_layer2
        delta_k = self.error_at_layer2(d_o_c,o)

        # 3. Compute the derivative of W (Written section, Part c).
        # Call: derivative_of_w
        layer1_out = self.fully_connected(X, self.U, self.e)

        b = self.sigmoid(layer1_out)
        dw = self.derivative_of_w(b,delta_k)

        # 4. Compute the derivative of f (Written section, Part d).
        # Call: derivative_of_f
        df  = self.derivative_of_f(delta_k)

        # 5. Compute the error at the first layer (Written section, Part e).
        # Call: error_at_layer1
        delta_j = self.error_at_layer1(delta_k,self.W,b)


        # 6. Compute the derivative of U (Written section, Part f).
        # Call: derivative_of_u
        du = self.derivative_of_u(X,delta_j)

        # 7. Compute the derivative of e (Written section, Part g).
        # Call: derivative_of_e
        de = self.derivative_of_e(delta_j)

        # 8. Update the parameters
        self.W = self.W - dw
        self.U  = self.U - du
        self.e = self.e - de
        self.f = self.f - df



    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
        print("tttt:", np.shape(X))
        print("oooo:",np.shape(o))


def main():
    """ Main function """
    #np.random.seed(101)
    # generate random input data of dimension (batchSize, inputSize).
    a = np.random.randint(0, high=10, size=[3,2], dtype='l')

    # generate random ground truth.
    t = np.random.randint(0, high=100, size=[3,1], dtype='l')

    # scale the input and output data.
    a = a/np.amax(a, axis=0)
    t = t/100

    # create an instance of Neural_Network.
    NN = Neural_Network()
    for i in range(num_itr):
        print("Input: \n" + str(a))
        print("Actual Output: \n" + str(t))
        print("Predicted Output: \n" + str(NN.forward(a)))
        print("Loss: \n" + str(np.mean(np.square(t - NN.forward(a)))))
        print("\n")

        NN.train(a, t)


if __name__ == "__main__":
    main()
