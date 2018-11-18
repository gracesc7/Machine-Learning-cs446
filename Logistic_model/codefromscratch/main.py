"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = None
max_iters = None

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset','indexing.txt')
    A,T = read_dataset('/home/szhan114/cs446sp_2018/hw/mp3/data/trainset','indexing.txt')
    #print("A shape:", np.shape(A))
    #print("T shape:", np.shape(T))
    # Initialize model.
    ndim = np.shape(A)[1]-1
    N = np.shape(A)[0]
    #print("N",N)
    model = LogisticModel(ndim, 'zeros')

    # Train model via gradient descent.s
    model.fit(T, A, 0.0001, 1000)
    prediction = model.classify(A)
    accuracy = 0
    #print("pridiction dimension:", np.shape(prediction))
    for i in range(N):
        print("predict:",prediction[i]," ",T[i])
        if(T[i] == prediction[i]):
            accuracy += 1
            print("yes")
    print("Accuracy:",accuracy)
    model.save_model("trained weights.np")

    # Save trained model to 'trained_weights.np'

    # Load trained model from 'trained_weights.np'

    # Try all other methods: forward, backward, classify, compute accuracy

    pass
