"""Input and output helpers to load in data.
"""

import pickle
import numpy as np
from numpy import genfromtxt


def read_dataset(input_file_path):
    """Read input file in csv format from file.
    In this csv, each row is an example, stored in the following format.
    label, pixel1, pixel2, pixel3...

    Args:
        input_file_path(str): Path to the csv file.
    Returns:
        (1) label (np.ndarray): Array of dimension (N,) containing the label.
        (2) feature (np.ndarray): Array of dimension (N, ndims) containing the
        images.
    """
    # Imeplemntation here.
    f = open(input_file_path,'r')
    file_read = f.readlines()
    num_data = len(file_read)
    labels = np.zeros([num_data,])
    feature_list = []
    for i in range(len(file_read)):
        feature = []
        img = file_read[i].strip('\n')
        img = img.split(',')
        img = [float(j) for j in img]
        labels[i] = img[0]
        for k in range(1,len(img)):
            feature.append(img[k])
        feature_list.append(feature)
    features= np.array(feature_list)
    #print(np.shape(features))
    return labels, features
