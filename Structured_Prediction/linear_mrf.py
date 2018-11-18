import tensorflow as tf
import numpy as np
from collections import defaultdict


class LinearMRF(object):
    '''Class implementing the MRF used for denoising'''

    def __init__(self, width, height):
        '''Initializes the MRF model.

        Specifically, the model used is a grid graph, i.e. there is a
        node for every pixel in an image and a pair connecting every pair
        of adjacent pixels.

        The nodes are numbered sequentially, starting from the upper right
        corner of the image, first increasing to the right and then downward.
        For example, here are the numberings of the nodes of a 3x4 graph:
        0  1  2  3
        4  5  6  7
        8  9  10 11

        Pairs are represented as tuples (m, n), where m and n are the
        indices of the two nodes. For example, in the above graph,
        (5, 9) and (1, 2) are both pairs.

        After initialization, this class contains the following fields:
            width(int): the width in pixels of the images represented
            height(int): the height in pixels of the images represented
            unary_weight(tf.Variable): the linear weight for unary potentials
            pairwise_weight(tf.Variable): the linear weight for pairwise
                potentials
            pairs(list of tuples): represents the edges in the MRF. Each entry
                is a tuple containing two integers, each representing a node
            pair_inds(dict): provides the index within the list of pairs
                for a given pair
            neighbors(dict): the keys are nodes (integers), and the values
                are a list of integers representing the nodes adjacent to
                the specified node in the MRF
            pairwise_features(np.array): The pairwise features for the model
        '''
        self.width = width
        self.height = height
        self.unary_weight = tf.Variable(1.)
        self.pairwise_weight = tf.Variable(1.)

        self.pairs = []
        self.pair_inds = {}
        self.neighbors = defaultdict(list)
        i = 0


        #pair_inds: dictionary, key is a pair, value is i, ith to be inserted in dict
        # Initialize neighbors, pairs, and pair_inds
        for row in range(height):
            for col in range(width):
                ind1 = row*width + col

                if row > 0:
                    ind2 = (row-1)*width + col          #vertical neighbors
                    self.pairs.append((ind2, ind1))
                    self.pair_inds[(ind2, ind1)] = i   #pair_inds dict: key-pair of node (1,2 or 2,1) share same i  value - ith inserted into dict
                    self.pair_inds[(ind1, ind2)] = i

                    self.neighbors[ind1].append(ind2)
                    self.neighbors[ind2].append(ind1)

                    i += 1
                if col > 0:
                    ind2 = row*width + (col-1)          #horizontal neighbors
                    self.pairs.append((ind2, ind1))
                    self.pair_inds[(ind2, ind1)] = i
                    self.pair_inds[(ind1, ind2)] = i

                    self.neighbors[ind1].append(ind2)
                    self.neighbors[ind2].append(ind1)

                    i += 1
        #print("self.pairs shape:",np.shape(self.pairs))
        #print("self.pair_inds:", len(self.pair_inds))

        # pre-compute pair features, since they do not depend on the image
        self.pairwise_features = self.get_pairwise_features()

    def get_unary_features(self, img):
        '''Calculates the full matrix of unary features for a provided
        set of values (which can be either the noisy observations or the
        true image)

        Recall that, for a given node observation x_i and possible assignment
        y_i to that node, the feature function you should calculate is

        f_unary(x_i, y_i) = 1[x_i == y_i]

        where 1[] is the indicator function that equals 1 if the argument is
        true and 0 if the argument is false.

        As mentioned, this calculates the full matrix of unary features - you
        should use the following index scheme:

        result[i, 0] is the feature value for node x_i when y_i = 0
        result[i, 1] is the feature value for node x_i when y_i = 1

        Args:
            img(np.array): An array of length (width x height) representing
              the observations of an image
        Returns:
            (np.array): An array of size (width x height, 2) containing the
              features for an image
        '''
        #result(i,0) = x_i and y_i =0's f_unary result(i,1) = x_i and y_i = 1's f_unary
        #######################
        #IMPLEMENT THIS METHOD#
        #######################
        #print("img len:", len(img) == self.width*self.height)
        result = np.zeros((len(img),2))
        #print("len(img) =? h*w: ", len(img) == self.height *self.width)
        for i in range(len(img)):
                if(img[i] == 0):
                        #result[i][0] = 1
                        #result[i][1] = 0
                        result[i,0] = 1
                        result[i,1] = 0
                if(img[i] == 1):
                        #result[i][1] = 1
                        #result[i][0] = 0
                        result[i,0] = 0
                        result[i,1] = 1
        #print("unary feature shape:", np.shape(result))
        return result

    def get_pairwise_features(self):
        '''Calculates the full matrix of pairwise features.

        Recall that, for a given set of possible assignments y_i and y_j to
        a pair of nodes (i, j) in the graph, the feature function you should
        calculate is

        f_pairwise(y_i, y_j) = 1[y_i == y_j]

        where 1[] is the indicator function that equals 1 if the argument is
        true and 0 if the argument is false.

        As mentioned, this calculates the full matrix of pairwise features -
        you should use the following index scheme, where i is the index for
        pair (m, n) as found in self.pair_inds (and m < n):

        result[i, 0] is the feature value for y_m = 0, y_n = 0
        result[i, 1] is the feature value for y_m = 0, y_n = 1
        result[i, 2] is the feature value for y_m = 1, y_n = 0
        result[i, 3] is the feature value for y_m = 1, y_n = 1

        Returns:
            (np.array): An array of size (len(pairs), 4) containing the
              pairwise features for an image
        '''
        #######################
        #IMPLEMENT THIS METHOD#
        #######################
        #self.pairs is a list of tuple
        #UNSURE for y_m =0, what is value for [i,2]and [i,3]
        result = np.zeros((len(self.pairs),4))
        #UNSURE or can use pair_inds function for i
        i = 0
        for i in range(len(self.pairs)):  #!!!!!!!!!!!!!DUMB MISTAKE HERE
            #i = self.pair_inds[(pair)]
            #if(pair[0]<pair[1]):
            #print("pair 0:", pair[0])
            #if(pair[0]==0):
            result[i,0] = 1
            result[i,1] = 0
            result[i,2] = 0
            result[i,3] = 1
            #if(pair[0] ==1):
            '''
                result[i][2] = 0
                result[i][3] = 1
                result[i][0] = 1
                result[i][1] = 0
            i += 1
            '''
        #print("pairwise features shape:",np.shape(result))
        return result

    def calculate_unary_potentials(self, unary_features):
        '''Calculates the full matrix of unary potentials for a provided
        matrix of unary features.

        Recall that, for a given node observation x_i and an assignment y_i to
        that node, the potential function you should calculate is

        phi_i(x_i, y_i) = w_unary * f_unary(x_i, y_i)

        where f_unary(x_i, y_i) is the value of the feature function for a
        given node/assignment

        Args:
            unary_features(np.array): a (height * width, 2)-sized matrix
              containing the features for a noisy sample (see
              get_unary_features for more details)
        Returns:
            (tf.Tensor): The unary potentials, which is a rank-2 tensor of the
              same size as the unary_features.
        '''
        #######################
        #IMPLEMENT THIS METHOD#
        #######################
        #    self.unary_weight = tf.Variable(1.)
        #print("unary weight shape:", tf.shape(self.unary_weight))
        #sess = tf.Session()
        #sess.run(tf.global_variables_initializer())
        '''
        u_feature = tf.convert_to_tensor(unary_features)
        u_feature = tf.cast(u_feature,tf.float32)
        '''
        #print("test:u_feature ",u_feature)
        #print("test:self.unary_weight: ", self.unary_weight)
        #result = tf.scalar_mul(self.unary_weight,u_feature)  #EDIT
        '''
        result = tf.scalar_mul(self.unary_weight,u_feature)
        '''

        result = self.unary_weight * unary_features

        #print("wei0:",self.unary_weight)
        #print("test:result of tensor matmul: ", result)
        #output = sess.run(result)
        return result

    def calculate_pairwise_potentials(self, pairwise_features):
        '''Calculates the full matrix of pairwise potentials for a provided
        matrix of pairwise features.

        Recall that, for a given pair of assignments y_i and y_j to nodes i
        and j, the potential function you should calculate is

        phi_ij(y_i, y_j) = w_pairwise * f_pairwise(y_i, y_j)

        where f_pairwise(y_i, y_j) is the value of the pairwise feature
        function for a given node/assignment

        Args:
            pairwise_features(np.array): a (len(pairs), 4)-sized matrix
              containing the pairwise features (see get_pairwise_features
              for more details)
        Returns:
            (tf.Tensor): The pairwise potentials, which is a rank-2 tensor of
              the same size as the pairwise_features
        '''
        #######################
        #IMPLEMENT THIS METHOD#
        #######################
        #print("unary weight shape:", tf.shape(self.unary_weight))
        #sess = tf.Session()
        #sess.run(tf.global_variables_initializer())
        '''
        p_feature = tf.convert_to_tensor(pairwise_features)
        p_feature = tf.cast(p_feature,tf.float32)
        '''
        #result = tf.scalar_mul(self.pairwise_weight,p_feature)
        #print("wei:", self.pairwise_weight)
        '''
        result = tf.scalar_mul(self.pairwise_weight,p_feature)
        '''
        result = self.pairwise_weight * pairwise_features

        #output = sess.run(result)
        #print("test: output ", output)
        return result


    def build_training_obj(self, img_features, unary_beliefs, pair_beliefs,
                           unary_potentials, pairwise_potentials):
        '''Builds the training objective, as specified in the handout.

        Hint: the image features can be thought of as a "correct" set of
        beliefs for that image

        Args:
            img_features(np.array): The unary feature representation of the
                true image (as returned by get_unary_features)
            unary_beliefs(list(tf.Tensor)): A list of the unary beliefs
                for each of the noisy samples
            pair_beliefs(tf.Tensor): A list of the pairwise beliefs for each
                of the noisy samples
            unary_potentials(list(tf.Tensor)): A list of the unary potentials
                for each of the noisy samples. Each entry is a rank-2 tensor
                of size (height x width, 2)
            pairwise_potentials(tf.Tensor): The pairwise potentials, which is
                a rank-2 tensor of size (height x width, 4)
        Returns:
            (tf.Tensor): the training objective, which is a rank-0 tensor
        '''
        #phi: potential function
        #unary_beliefs: a list of (36864,2)
        obj = 0
        #sess = tf.Session()
        #sess.run(tf.global_variables_initializer())
        #print("test:unary_beliefs: ",unary_beliefs )

        #print("type of unary potentials:", np.shape(unary_potentials[0]))
        #print("*************************************")
        #print("shape of tensor:",unary_beliefs[0].get_shape())
        #tf_shape = unary_beliefs[0].get_shape()
        '''
        u_obj = tf.constant(0.0, tf.float32)
        p_obj = tf.constant(0.0,tf.float32)
        '''
        #b = u_obj +
        #print("type of poten: ", unary_potentials)
        #print("type of be: ", unary_beliefs)
        #print("pair be: ", pair_beliefs )
        #print("pair poten:", pairwise_potentials)
        '''
        for i in range(len(unary_beliefs)):
            #unary_potentials[i] = tf.convert_to_tensor(unary_potentials[i],tf.float32)
            u = tf.multiply(unary_beliefs[i], unary_potentials[i])
            u_sum = tf.reduce_sum(u)
            u_sum = tf.cast(u_sum,tf.float32)
            #print("usum: ",i, sess.run(u_sum))
            u_obj = tf.add(u_obj,u_sum)
        '''
        #print("pair potential:", pairwise_potentials)
        #pairwise_potentials = tf.convert_to_tensor(pairwise_potentials,tf.float32)
        #p_obj = tf.matmul(pair_beliefs,pairwise_potentials)
        #print("p_obj: ", p_obj)
        '''
        p_obj = tf.constant(0.0,tf.float32)
        print("pairwise pote:",pairwise_potentials)
        print("pair beli:", pair_beliefs)
        for j in range(len(pair_beliefs)):
            p = tf.multiply(pair_beliefs[j], pairwise_potentials)
            p_sum = tf.reduce_sum(p)
            p_sum = tf.cast(p_sum,tf.float32)
            p_obj = tf.add(p_obj,p_sum)
        '''
        #img_features = tf.cast(img_features,tf.float32)
        u_beliefs = [tf.cast(i,tf.float32) for i in unary_beliefs]
        p_beliefs = [tf.cast(i,tf.float32) for i in pair_beliefs]
        u_potential = [tf.cast(i,tf.float32) for i in unary_potentials]
        p_potential = tf.cast(pairwise_potentials,tf.float32)
        obj += tf.reduce_sum(tf.multiply(u_beliefs,u_potential))
        obj += tf.reduce_sum(tf.multiply(p_beliefs,p_potential))
        #obj -= tf.reduce_sum(tf.multiply(u_potential,img_features))
        obj -= tf.reduce_sum(self.calculate_unary_potentials(img_features))


        #print("pairwise belief:", pair_beliefs)
        #print("pairwise_potentials: ", pairwise_potentials)
        #for j in range(len(pair_beliefs)):

        #print("u_obj:", u_obj)
        #a = sess.run(u_obj)
        #print("hahaha", a)

        #m = tf.placeholder(tf.float32)
        #n = tf.placeholder(tf.float32)
        #m = unary_beliefs[0]
        #n = unary_beliefs[1]
        #a_sum = tf.placeholder(tf.float32)
        #a = m+n
        #a_sum = tf.reduce_sum(a)
        #print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
        #print("aaa:", a, "bbbbb", a_sum)
        #print("aaaaaaa:", sess.run(a,{m:unary_beliefs[0],n:unary_beliefs[1]}))

        #unary_potentials = tf.convert_to_tensor(unary_potentials)
        #print("test:unary_potentials ", unary_potentials)
        #u = tf.multiply(unary_beliefs,unary_potentials)
        #u_obj = tf.reduce_sum(u)
        #p = tf.multiply(pair_beliefs,pairwise_potentials)
        #p_obj = tf.reduce_sum(p)
        #up_obj = tf.add(u_obj,p_obj)
        #up_obj = u_obj + p_obj
        #true_u_potential = self.calculate_unary_potentials(img_features)
        #true_u_potential = tf.multiply(unary_potentials,img_features)    #WHY

        true_p =tf.cast(self.get_pairwise_beliefs(img_features), dtype = tf.float32)
        obj -= tf.reduce_sum(tf.multiply(true_p,p_potential)) * len(unary_beliefs)

        #F_u = tf.reduce_sum(true_u_potential)  #UNSURE
        #F_p = tf.reduce_sum(true_p_potential)
        #F = tf.add(F_u,F_p)
        #print("test:up_obj, F ", up_obj, F)
        #up_obj = tf.cast(up_obj,tf.float32)
        #obj = tf.cast(obj,tf.float32)
        #obj = obj - F
        #print("objjjjjjj:", obj)
        #print("obj",obj)
        #######################
        #IMPLEMENT THIS METHOD#
        #######################
        return obj

    def train(self, original_img, noisy_samples, lr, num_epochs,
              convergence_margin):
        '''Trains the model using the provided data and training parameters.

        Args:
            original_img(np.array): The true, denoised image
            noisy_samples(list(np.array)): Noisy samples of the true image
            lr(float): The learning rate for gradient descent
            num_epochs(int): The number of training iterations
            convergence_margin(float): The convergence margin for inference
                (see run_greedy_inference for more details)
        '''

        # Initialize placeholders for beliefs
        unary_belief_placeholders = []
        pairwise_belief_placeholders = []
        for i in range(len(noisy_samples)):
            unary_belief_placeholders.append(tf.placeholder(tf.float32,
                                             [self.height*self.width, 2]))
            pairwise_belief_placeholders.append(tf.placeholder(tf.float32,
                                                [len(self.pairs), 4]))

        # Compute features for original image and noisy samples
        img_features = self.get_unary_features(original_img)
        noisy_features = [self.get_unary_features(noisy)
                          for noisy in noisy_samples]

        # Compute initial beliefs. We initialize them to be identical to the
        # noisy features (meaning the beliefs are set such that the model
        # believes the noisy observations to be correct
        unary_beliefs = []
        pairwise_beliefs = []
        unary_beliefs = [feat.copy() for feat in noisy_features]
        pairwise_beliefs = [self.get_pairwise_beliefs(belief)
                            for belief in unary_beliefs]

        # Build the computation graph for training
        unary_potentials = [self.calculate_unary_potentials(feat)
                            for feat in noisy_features]
        pairwise_potentials = self.calculate_pairwise_potentials(
                self.pairwise_features)
        train_obj = self.build_training_obj(img_features,
                                            unary_belief_placeholders,
                                            pairwise_belief_placeholders,
                                            unary_potentials,
                                            pairwise_potentials)
        #print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
        optimizer = tf.train.GradientDescentOptimizer(lr)

        train = optimizer.minimize(train_obj)
        #print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):
                print("EPOCH %d" % (epoch+1))

                # First: Calculate Potentials
                nodes = [pairwise_potentials] + unary_potentials
                results = sess.run(nodes)
                pairwise_pot_result = results[0]
                unary_pot_result = results[1:]

                # Second: Run Inference
                unary_beliefs, pairwise_beliefs = self.run_greedy_inference(
                        unary_beliefs, unary_pot_result, pairwise_pot_result,
                        convergence_margin)

                score = 0
                for unary_belief in unary_beliefs:
                    score += np.sum(unary_belief != img_features) /\
                            (len(unary_belief) * 2)  #EDIT
                score /= len(unary_beliefs)
                print("CURRENT SCORE: ", score)
                print("WEIGHTS: ")
                print(sess.run(self.unary_weight))
                print(sess.run(self.pairwise_weight))

                # Third: Update model parameters based on current beliefs
                feed_dict = {}
                for belief, placeholder in zip(unary_beliefs,
                                               unary_belief_placeholders):
                    feed_dict[placeholder] = belief
                for belief, placeholder in zip(pairwise_beliefs,
                                               pairwise_belief_placeholders):
                    feed_dict[placeholder] = belief
                nodes = [train, pairwise_potentials] + unary_potentials
                results = sess.run(nodes, feed_dict)

    def test(self, noisy_samples, convergence_margin):
        '''Given a list of noisy samples of an image, produce denoised
        versions of that image.

        Args:
            noisy_samples(list(np.array)): A list of noisy samples of an image
            convergence_margin(float): The convergence margin for inference
                (see run_greedy_inference for more details)
        Returns:
            (list(np.array)): The denoised images
        '''

        # Initialize Beliefs
        unary_beliefs = []
        noisy_features = [self.get_unary_features(noisy)
                          for noisy in noisy_samples]
        unary_beliefs = [feat.copy() for feat in noisy_features]

        unary_potentials = [self.calculate_unary_potentials(feat)
                            for feat in noisy_features]
        pairwise_potentials = self.calculate_pairwise_potentials(
                self.pairwise_features)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Calculate potentials
            nodes = [pairwise_potentials] + unary_potentials
            results = sess.run(nodes)
            pairwise_pot_result = results[0]
            unary_pot_result = results[1:]

        unary_beliefs, pairwise_beliefs = self.run_greedy_inference(
                unary_beliefs, unary_pot_result, pairwise_pot_result,
                convergence_margin)

        denoised_imgs = [self.beliefs2img(unary_belief)
                         for unary_belief in unary_beliefs]
        return denoised_imgs

    def beliefs2img(self, unary_belief):
        '''Converts a provided set of beliefs into the correct format for
        an image by setting each pixel to the value that has belief 1
        '''
        result = np.empty(len(unary_belief))
        for i in range(len(unary_belief)):
            if unary_belief[i, 0] == 1:
                result[i] = 0
            else:
                result[i] = 1
        return result.reshape((self.height, self.width))

    def run_greedy_inference(self, unary_beliefs, unary_pots, pairwise_pots,
                             convergence_margin):
        '''Runs our greedy inference procedure on the provided lists of
        beliefs/potentials. Note that we run inference for a maximum of 10
        iterations per image.
        '''
        new_unary_beliefs = []
        new_pairwise_beliefs = []

        for unary_belief, unary_pot in zip(unary_beliefs, unary_pots):
            itr = 0
            converged = False

            unary_belief = unary_belief.copy()

            while not converged:
                if itr > 10:
                    break

                itr += 1

                new_unary_belief = self.inference_itr(unary_belief, unary_pot,
                                                      pairwise_pots)
                converged = self.check_convergence(new_unary_belief,
                                                   unary_belief,
                                                   convergence_margin)

                unary_belief = new_unary_belief

            pairwise_belief = self.get_pairwise_beliefs(unary_belief)
            new_unary_beliefs.append(unary_belief)
            new_pairwise_beliefs.append(pairwise_belief)
        return new_unary_beliefs, new_pairwise_beliefs

    def inference_itr(self, unary_beliefs, unary_pots, pairwise_pots):
        '''Run a single iteration of inference with the provided beliefs
        and potentials.

        Here, inference should be implemented as a simple (randomized) greedy
        algorithm. The steps are as follows:
        1. Determine a random order of nodes
        2. For each node in the graph:
            a. Calculate the scores (using calculate_local_score) for that
               node having assignment 0 and 1
            b. Set the belief for the assignment having the larger score to 1
               and that for the assignment having the smaller score to 0

        Args:
            unary_beliefs(np.array): The input set of beliefs, having shape
            (width x height, 2)
            unary_pots(np.array): The unary potentials for the image, having
              shape (width x height, 2)
            pairwise_pots(np.array): The pairwise potentials for the image,
              having shape (len(self.pairs), 4)
        Returns:
            (np.array): The new set of unary beliefs, having the same shape
              as the input set of unary beliefs.   (widthxheight, 2s)
        '''
        #print("test:unary_beliefs shape:", np.shape(unary_beliefs))
        num_node = np.shape(unary_beliefs)[0]
        unary_beliefs = unary_beliefs.copy()
        nodes = np.arange(0,num_node)
        #print("test: nodes shape", np.shape(nodes))
        np.random.shuffle(nodes)   #UNSURE
        print(nodes)
        for i in range(num_node):
            #node = nodes[i]
            score_0 = self.calculate_local_score(i,0,unary_beliefs,unary_pots,pairwise_pots)
            score_1 = self.calculate_local_score(i,1,unary_beliefs,unary_pots,pairwise_pots)
            if(score_0 > score_1):  #UNSURE > = ?
                unary_beliefs[i,0] = 1
                unary_beliefs[i,1] = 0
            else:
                unary_beliefs[i,1] =1
                unary_beliefs[i,0] = 0
        #######################
        #IMPLEMENT THIS METHOD#
        #######################
        return unary_beliefs

    def calculate_local_score(self, node, assignment, unary_beliefs,
                              unary_potentials, pairwise_potentials):
        '''Calculates the score of a "patch" surrounding the specified node,
        assuming that node has the specified assignment, given the current
        beliefs for the assignments of values to the pixels in the image

        This score consists of the sum of the unary potential for this node
        given this assignment, plus the pairwise potentials of all pairs
        that include this node given the assignment specified for this node
        and the assignment for the other nodes specified by the provided
        unary beliefs.

        Args:
            node(int): The node whose patch is being scored
            assignment(int): The assignment that should be considered for
              the node (either 0 or 1)
            unary_beliefs(np.array): The current set of unary beliefs for
              the image, having shape (width x height, 2)
            unary_potentials(np.array): The current set of unary potentials
              for the image, having shape (width x height, 2)
            pairwise_potentials(np.array): The current set of pairwise
              potentials for the image, having shape (len(self.pairs), 4)
        Returns:
            (float): The score of the patch
        '''
        #sess = tf.Session()
        score = 0.0
        #sum_p_potential = 0
        neighbors = self.neighbors[node]   #values are location idx of the neighbor
        #n_pair_list = []   #pair the current node with its neighbors, list of tuple
        #pair_idx_list = [] #a list of index of each pair in the pairlist, for the purpose to access pairwise potential
        '''
        for n in neighbors:
            pair = (node,n)
            #n_pair_list.append(pair)
            i = self.pair_inds[pair]  #index of each pair in the pairlist, for the purpose to access pairwise potential
            #print("test:pair: ", pair, "i ", i)
            #print("test:unary_beliefs[n,0]: ",unary_beliefs[n,0])
            b_0 = unary_beliefs[n,0]  # b_1 = 1-b_0
            if(assignment ==0 and b_0 ==1):  #(node assignment,neighbor belief) = (0,0) = (y_m,y_n)
                sum_p_potential += pairwise_potentials[i,0]
            if(assignment ==0 and b_0 ==0):  #(0,1)
                sum_p_potential += pairwise_potentials[i,1]
            if(assignment ==1 and b_0 ==1): #(1,0)
                sum_p_potential += pairwise_potentials[i,2]
            if(assignment ==1 and b_0 ==0):#(1,1)
                sum_p_potential += pairwise_potentials[i,3]
        '''
        for n in neighbors:
            idx = self.pair_inds[(n,node)]
            if(n<node):
                score += pairwise_potentials[idx][2*int(unary_beliefs[n,1])+int(assignment)]
            else:
                score += pairwise_potentials[idx][2*int(assignment) + int(unary_beliefs[n,1])]

        if(assignment ==0):
            u_potential = unary_potentials[node][0]
        else:
            u_potential = unary_potentials[node][1]
        #u_p = sess.run(u_potential)
        #sum_p = sess.run(sum_p_potential)
        score += u_potential
        #print("local score: ", score )
        #print("type of score:", type(score))
        #######################
        #IMPLEMENT THIS METHOD#
        #######################
        #Since pairwise potentials matrix for a certain pair has 4 entries, we need the belief term to determine the exact entry we're going to use.

        return score

    def check_convergence(self, new_unary_beliefs, old_unary_beliefs,
                          convergence_margin):
        '''Given two sets of unary beliefs, determine if inference has
        converged.

        Convergence occurs when the percentage of nodes in the graph whose
        beliefs have changed is less than the provided margin.

        Args:
            new_unary_beliefs(np.array): One set of unary beliefs, having
              the same shape as elsewhere in the code
            old_unary_beliefs(np.array): Another set of unary beliefs, having
              the same shape as elsewhere in the code
            convergence_margin(float): the threshold determining convergence.
              This should be a number between 0 and 1
        Returns:
            (bool): whether inference has converged
        '''
        count = 0
        for i in range(np.shape(new_unary_beliefs)[0]):
            if(new_unary_beliefs[i,0] != old_unary_beliefs[i,0]):
                count += 1
            if(new_unary_beliefs[i,1] != old_unary_beliefs[i,1]):
                count += 1
        p_change = count/(2*np.shape(new_unary_beliefs)[0]) #UNSURE type of p_change
        #print("percentage of change:", p_change)
        if(p_change < convergence_margin):
            result = True
        else:
            result = False

        #######################
        #IMPLEMENT THIS METHOD#
        #######################

        return result

    def get_pairwise_beliefs(self, unary_beliefs):
        '''Generates the appropriate pairwise beliefs for a specified set of
        unary beliefs.

        Due to the fact that all of the unary beliefs for this inference
        implementation are either 0 or 1, the pairwise beliefs are a
        simple deterministic function of the unary beliefs.

        Specifically, given a pair of nodes (m, n), the pairwise belief
        for assignment (y_m, y_n) = 1 iff the unary belief for node m with
        assignment y_m is 1 and the unary belief for node n with assignment
        y_n is 1.

        Args:
            unary_beliefs(np.array): The set of unary beliefs for a noisy
              sample. This array has shape (width x height, 2)
        Returns:
            (np.array): The set of pairwise beliefs. This array should have
              shape (len(self.pairs), 4) and is indexed the same way as
              specified in get_pairwise_features.
        '''
        result = np.zeros((len(self.pairs), 4))
        for i in range(len(self.pairs)):
            node0 = self.pairs[i][0]
            node1 = self.pairs[i][1]  #UNSURE
            if(unary_beliefs[node0,0] ==1 and unary_beliefs[node1,0] ==1):  # if node0 is 0, node1 is 0
                result[i,0] = 1
            if(unary_beliefs[node0,0] ==1 and unary_beliefs[node1,1] ==1): #if node0 is 0, node1 is 1
                result[i,1] = 1
            if(unary_beliefs[node0,1] ==1 and unary_beliefs[node1,0] ==1): #if node1 is 1, node1 is 0
                result[i,2] = 1
            if(unary_beliefs[node0,1] ==1 and unary_beliefs[node1,1] ==1): #if node1 is 1, node1 is 1
                result[i,3] =1
        #print("result:",result)
        #######################
        #IMPLEMENT THIS METHOD#
        #######################
        return result
