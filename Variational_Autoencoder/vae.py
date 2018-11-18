"""Variation autoencoder."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers
from tensorflow.contrib.slim import fully_connected


class VariationalAutoencoder(object):
    """Varational Autoencoder.
    """
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a VAE. (**Do not change this function**)

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Create session
        self.session = tf.Session()
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Build graph.
        self.z_mean, self.z_log_var = self._encoder(self.x_placeholder)
        self.z = self._sample_z(self.z_mean, self.z_log_var)
        self.outputs_tensor = self._decoder(self.z)

        # Setup loss tensor, predict_tensor, update_op_tensor
        self.loss_tensor = self.loss(self.outputs_tensor, self.x_placeholder,
                                     self.z_mean, self.z_log_var)

        self.update_op_tensor = self.update_op(self.loss_tensor,
                                               self.learning_rate_placeholder)

        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())

    def _sample_z(self, z_mean, z_log_var):
        """Samples z using reparametrization trick.

        Args:
            z_mean (tf.Tensor): The latent mean,
                tensor of dimension (None, _nlatent)
            z_log_var (tf.Tensor): The latent log variance,
                tensor of dimension (None, _nlatent)
        Returns:
            z (tf.Tensor): Random sampled z of dimension (None, _nlatent)
        """
        '''
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),
                               mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        '''
        #shape = tf.Variable(self._nlatent)
        epsilon = tf.random_normal(shape = tf.shape(z_log_var), dtype = tf.float32)
        z = z_mean + tf.sqrt(tf.exp(z_log_var)) * epsilon
        #z = None
        ####### Implementation Here ######
        print('sample_z',z)   #shape (?,2) correct
        '''
        shape = tf.Variable(self._nlatent)
        z = tf.add(z_mean, tf.matmul(z_log_var,tf.random_normal(shape)))

        print('shape of z',np.shape(z))
        '''
        #pass
        return z

    def _encoder(self, x):
        """Encoder block of the network.

        Builds a two layer network of fully connected layers, with 100 nodes,
        then 50 nodes, and outputs two branches each with _nlatent nodes
        representing z_mean and z_log_var. Network illustrated below:

                             |-> _nlatent (z_mean)
        Input --> 100 --> 50 -
                             |-> _nlatent (z_log_var)

        Use activation of tf.nn.softplus for hidden layers.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, _ndims).
        Returns:
            z_mean(tf.Tensor): The latent mean, tensor of dimension
                (None, _nlatent).
            z_log_var(tf.Tensor): The latent log variance, tensor of dimension
                (None, _nlatent).
        """


        layer_100 = tf.contrib.slim.fully_connected(inputs = x,num_outputs = 100, activation_fn=tf.nn.softplus)
        #out_100 = tf.nn.softplus(layer_100)
        #print('out_100',out_100)
        print('layer_100',layer_100)
        layer_50 = tf.contrib.slim.fully_connected(inputs = layer_100, num_outputs = 50, activation_fn = tf.nn.softplus)
        #out_50 = tf.nn.softplus(layer_50)
        #print('out_50',out_50)
        print('layer_50',layer_50)
        z_mean  = tf.contrib.slim.fully_connected(inputs = layer_50, num_outputs = 2, activation_fn = None)

        z_log_var = tf.contrib.slim.fully_connected(inputs = layer_50,num_outputs = 2, activation_fn = None)

        print('z_mean', z_mean) #shape (?,2) correct
        print('z_log_var', z_log_var)   #shape (?,2) correct
        ####### Implementation Here ######
        return z_mean, z_log_var
        '''
        pass
        '''

    def _decoder(self, z):
        """From a sampled z, decode back into image.

        Builds a three layer network of fully connected layers,
        with 50, 100, _ndims nodes.

        z (_nlatent) --> 50 --> 100 --> _ndims.

        Use activation of tf.nn.softplus for hidden layers.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, _nlatent).
        Returns:
            f(tf.Tensor): Decoded features, tensor of dimension (None, _ndims).
        """
        layer_50 = tf.contrib.slim.fully_connected(inputs = z, num_outputs = 50, activation_fn = tf.nn.softplus)
        layer_100 = tf.contrib.slim.fully_connected(inputs = layer_50, num_outputs = 100, activation_fn = tf.nn.softplus)
        f = tf.contrib.slim.fully_connected(inputs = layer_100, num_outputs = self._ndims, activation_fn = tf.nn.sigmoid)
        #f = None
        ####### Implementation Here ######
        print('f decoder', f)     #shape (?,784)  correct
        #pass

        return f

    def _latent_loss(self, z_mean, z_log_var):
        """Constructs the latent loss.

        Args:
            z_mean(tf.Tensor): Tensor of dimension (None, _nlatent)
            z_log_var(tf.Tensor): Tensor of dimension (None, _nlatent)
            z_log_var: log(sigma^2)

        Returns:
            latent_loss(tf.Tensor): A scalar Tensor of dimension ()
                containing the latent loss.
        """

        #latent_loss = None
        # latent loss =  - D_KL
        '''
        sess = tf.Session()
        mean = sess.run(z_mean)
        print('mean',mean)
        print('var',var)
        var_diag = tf.diag_part(z_log_var)
        var = sess.run(var_diag)  #extract the digonal elements (log sigma^2)
        mu_sqr = tf.map_fn(lambda x:x*x, mean)   # np.array
        sigma_sqr = tf.map_fn(lambda y:2**y, var)  #np.array

        sigma_sqr_sum = tf.reduce_sum(sigma_sqr)
        sigma_log_sum = tf.reduce_sum(var)
        mu_sqr_sum = tf.reduce_sum(mu_sqr)

        loss = -sess.run(sigma_log_sum)[0]-1/2 + 1/2 * sess.run(sigma_sqr_sum)[0] + 1/2 * sess.run(mu_sqr_sum)[0]
        latent_loss = tf.Variable(loss,tf.float32)

        ####### Implementation Here ######
        #pass
        '''

        latent_loss = -1/2 * tf.reduce_sum(z_log_var +1 - tf.square(z_mean) - tf.exp(z_log_var),axis = 1)
        latent = tf.reduce_mean(latent_loss)
        print('latent',latent)
        #print('latentloss',latent_loss)
        #print('latent',latent)
        return latent

    def _reconstruction_loss(self, f, x_gt):
        """Constructs the reconstruction loss, assuming Gaussian distribution.

        Args:
            f(tf.Tensor): Predicted score for each example, dimension (None,
                _ndims).
            x_gt(tf.Tensor): Ground truth for each example, dimension (None,
                _ndims).
        Returns:
            recon_loss(tf.Tensor): A scalar Tensor for dimension ()
                containing the reconstruction loss.
        """


        #recon_loss = tf.losses.mean_squared_error(x_gt,f)

        recon_loss = tf.nn.l2_loss(f-x_gt)


        print('recon_loss',recon_loss)

        return recon_loss

    def loss(self, f, x_gt, z_mean, z_var):
        """Computes the total loss.

        Computes the sum of latent and reconstruction loss.

        Args:
            f (tf.Tensor): Decoded image for each example, dimension (None,
                _ndims).
            x_gt (tf.Tensor): Ground truth for each example, dimension (None,
                _ndims)
            z_mean (tf.Tensor): The latent mean,
                tensor of dimension (None, _nlatent)
            z_log_var (tf.Tensor): The latent log variance,
                tensor of dimension (None, _nlatent)

        Returns:
            total_loss: Tensor for dimension (). Sum of
                latent_loss and reconstruction loss.
        """

        total_loss = tf.add(self._reconstruction_loss(f,x_gt), self._latent_loss(z_mean,z_var))
        #loss = self.session.run(total_loss)
        print('total loss', total_loss)
        #print('total loss',loss)

        ####### Implementation Here ######
        #pass
        return total_loss

    def update_op(self, loss, learning_rate):
        """Creates the update optimizer.

        Use tf.train.AdamOptimizer to obtain the update op.

        Args:
            loss(tf.Tensor): Tensor of shape () containing the loss function.
            learning_rate(tf.Tensor): Tensor of shape (). Learning rate for
                gradient descent.
        Returns:
            train_op(tf.Operation): Update opt tensorflow operation.
        """
        #train_op = None
        ####### Implementation Here ######
        #pass
        train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
        return train_op

    def generate_samples(self, z_np):
        """Generates random samples from the provided z_np.

        Args:
            z_np(numpy.ndarray): Numpy array of dimension
                (batch_size, _nlatent).

        Returns:
            out(numpy.ndarray): The sampled images (numpy.ndarray) of
                dimension (batch_size, _ndims).
        """
        #out = None
        ####### Implementation Here ######
        #pass
        #print('z_np shape:', np.shape(z_np))
        #print('z',z_np)
        #print('num of ndims', self._ndims)
        #print('self.outputs_tensor',self.outputs_tensor)
        output = self.session.run(self.outputs_tensor,feed_dict = {self.z:z_np} )
        #print('output',output)
        #print('shape of batch:', np.shape(z_np)[0])
        #print('shape of output', np.shape(output))


        #pass
        return output
