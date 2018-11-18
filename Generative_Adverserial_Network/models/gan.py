"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers

class Gan(object):
    """Adversary based generator network.
    """
    def __init__(self, ndims=784, nlatent=10):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)
        g_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator')
        d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')

        # Add optimizers for appropriate variables
        self.update_op_tensor_g = self.update_op(self.g_loss,
                                                   self.learning_rate_placeholder,g_var)
        self.update_op_tensor_d = self.update_op(self.d_loss,
                                                   self.learning_rate_placeholder,d_var)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1).
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            layer_1= tf.contrib.slim.fully_connected(inputs = x, num_outputs = 151, activation_fn = tf.nn.relu)
            layer_2 = tf.contrib.slim.fully_connected(inputs = layer_1, num_outputs = 71,activation_fn = tf.nn.relu)
            y = tf.contrib.slim.fully_connected(inputs = layer_2, num_outputs = 1,activation_fn = None)
            print('y shape', tf.shape(y))
            return y


    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """

        l1 = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones(tf.shape(y)),logits = y)
        l2 = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros(tf.shape(y_hat)),logits = y_hat)
        l = tf.reduce_mean(l1+l2)
        print('_discriminator_loss shape,', tf.shape(l))
        return l


    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            layer_1= tf.contrib.slim.fully_connected(inputs = z, num_outputs = 151, activation_fn = tf.nn.relu)
            layer_2 = tf.contrib.slim.fully_connected(inputs = layer_1, num_outputs = 71,activation_fn = tf.nn.relu)
            x_hat = tf.contrib.slim.fully_connected(inputs = layer_2,num_outputs = self._ndims,activation_fn = None)
            print('x_hat shape:', tf.shape(x_hat))
            return x_hat


    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """

        l = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros(tf.shape(y_hat)),logits = y_hat ))
        print('generatorloss shape',tf.shape(l))
        return l

    def update_op(self, loss, learning_rate,var):
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
        train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss = loss,var_list = var )
        return train_op

'''
    def generate_samples(self, z_np):
        """Generates random samples from the provided z_np.

        Args:
            z_np(numpy.ndarray): Numpy array of dimension
                (batch_size, _nlatent).

        Returns:
            out(numpy.ndarray): The sampled images (numpy.ndarray) of
                dimension (batch_size, _ndims).
        """

        output = self.session.run(self.x_hat,feed_dict = {self.z_placeholder:z_np} )
        return output
        '''
