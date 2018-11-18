"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=10,
                 max_iter=100,
                 reg_covar=1e-6):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters

        #initialize self._mu in fit
        self._mu = np.ones((self._n_components,self._n_dims))
        #can use k-mean to initialize mu
        # Initialized with uniform distribution.   1/k
        self._pi = np.ones((self._n_components,1))
        for k in range(self._n_components):
            self._pi[k][0] = 1/self._n_components
        # np.array of size (n_components, 1)

        # Initialized with identity.
        self._sigma = np.array([np.identity(self._n_dims)*100] * self._n_components)  # np.array of size (n_components, n_dims, n_dims)
        #print(np.shape(self._sigma))

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        # first initilize self.mu
        '''
        for k in range(self._n_components):
            for j in range(self._n_dims):
                self._mu[k][j] = x[k][j]
        '''
        self._mu = 60 * np.random.rand(self._n_components,self._n_dims)

        for iter in range(self._max_iter):
            self._m_step(x,self._e_step(x))


        return 0

    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        return self.get_posterior(x)

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """


        # need to add reg_covar to the element in the diagonalof covariance matrix sigma
        # Update the parameters.
        N = np.shape(x)[0]
        for k in range(self._n_components):
            self._pi[k][0] = np.sum(z_ik[:,k])
        #mu:(n_components,n_dims)
        for k in range(self._n_components):
            for j in range(self._n_dims):
                self._mu[k][j] = np.matmul(z_ik[:,k],x[:,j].T)/self._pi[k][0]

        #self._sigma =                   #(n_components, n_dims, n_dims)
        self._pi = np.sum(z_ik,axis=0).reshape((self._n_components,1))
        self._mu = np.dot(z_ik.T,x)/(self._pi * N)
        sigma = []
        for i in range(self._n_components):
            vec1 = (z_ik[:,i].reshape((N,1)) * (x-self._mu[i])).T
            vec2 = x-self._mu[i]
            sigma.append(np.dot(vec1,vec2)/(self._pi[i]*N))
        self._sigma = np.array(sigma)
        self._pi = self._pi/N
        '''
        for j in range(self._n_dims):
            for k in range(self._n_components):
                mu = self._mu[k][j]
                x_mu = x[:,j] - mu
                sum_j = np.matmul(z_ik[:,k],x_mu * x_mu)
                sigma_k = sum_j/self._pi[k]
                self._sigma[k][j][j] = sigma_k
        self._pi = self._pi/N
        '''

        '''
        N = np.shape(x)[0]
        self._pi = np.sum(z_ik,axis=0).reshape((self._n_components,1))
        self._mu = np.dot(z_ik.T,x)/(self._pi * N)
        sigma = []
        for i in range(self._n_components):
            vec1 = (z_ik[:,i].reshape((N,1)) * (x-self._mu[i])).T
            vec2 = x-self._mu[i]
            sigma.append(np.dot(vec1,vec2)/(self._pi[i]*N))
        self._sigma = np.array(sigma)
        '''
        return 0

    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N,n_components).
        """
        ret = None
        ret = []
        N = np.shape(x)[0]
        prob = self._multivariate_gaussian(x,self._mu[0],self._sigma[0]).reshape(N,1)
        #print('shape of probk',np.shape(prob))
        for k in range(1,self._n_components):
            '''
            sig_k = self._sigma[k,:,:]+ self._reg_covar * np.identity(self._n_dims)
            mu_k = self._mu[k,:].flatten()
            '''
            prob_k = self._multivariate_gaussian(x,self._mu[k],self._sigma[k]).reshape(N,1)    # should be shape(N,1)
            '''
            prob_k = self._multivariate_gaussian(x,mu_k,sig_k).reshape(N,1)
            '''
            #print(np.shape(prob_k),np.shape(prob))
            #print('shape of probk',np.shape(prob_k))
            prob = np.concatenate((prob,prob_k),axis = 1)
        #print(np.shape(prob))
        #return np.array(ret)
        #print(np.shape(prob))

        return prob

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)
        p(x) = sum p(x|z)
        marginal = sum pi* p(x|z) = sum pi* conditional

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        N = np.shape(x)[0]
        P = self.get_conditional(x)
        marginal = []
        #print('P shape:', np.shape(P))
        #print('pi shape:',np.shape(self._pi))

        for i in range(N):
            sum_pxz = 0
            for k in range(self._n_components):
                #print(self._pi[k])
                #print(P[i][k])
                sum_pxz += self._pi[k][0] * P[i][k]
            marginal.append(sum_pxz)
        marginal = np.array(marginal)
        '''
        p_k = self._pi.reshape((1,self._n_components)) * self.conditional(x)
        marginal = np.sum(p_k,axis = 1)
        '''
        return marginal

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        N = np.shape(x)[0]
        z_ik = np.zeros((N,self._n_components))
        #marginal = self.get_marginals(x)     # (N,)
        #conditional = self.get_conditional(x)    #(N,n_components)
        '''
        margin_k = self._pi.reshape((1,self._n_components)) * self.get_conditional(x)
        marginal = np.sum(margin_k,axis=1 ).reshape((N,1))
        z_ik = margin_k/marginal
        '''
        for i in range(N):
            for k in range(self._n_components):
                z_ik[i][k]= (self._pi[k][0] * self.get_conditional(x)[i][k]+ self._reg_covar)/(self.get_marginals(x)[i]+self._reg_covar)

        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        self.fit(x)
        poster = self.get_posterior(x)   # shape: (N,n_components)
        #print((poster[:,0]))
        self.cluster_label_map = []    #length of n_components
        N = np.shape(x)[0]
        x_compo_list = []    #list of x's components based on highest posterior
        for i in range(N):
            max_compo = np.argmax(poster[i])
            x_compo_list.append(max_compo)
        x_compo = np.array(x_compo_list)  #arry form of x_compo_list
        for k in range(self._n_components):
            compo_idx = np.where(x_compo == k)
            if(np.shape(compo_idx)[1] == 0):
                self.cluster_label_map.append(4)
            #check if each component actually assign to any data x, if not, aobve
            else:
            #print('shape of compoidx',np.shape(compo_idx))
            #print('length of compodix',len(compo_idx))
                label_list = []
                for j in range(len(compo_idx)):
                    #print('y_combo',compo_idx[0][j])
                    #print('y',y[0])
                    label_list.append(y[compo_idx[0][j]])
                compo_label = np.argmax(np.bincount(label_list))
                self.cluster_label_map.append(compo_label)
        #pass
        return 0

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """

        z_ik = self.get_posterior(x)
        y_hat = []
        N = np.shape(x)[0]
        for i in range(N):
            max_compo = np.argmax(z_ik[i])
            y_hat.append(self.cluster_label_map[max_compo])

        return np.array(y_hat)
