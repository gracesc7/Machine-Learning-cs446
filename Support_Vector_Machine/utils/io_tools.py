"""Input and output helpers to load in data.
(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    data = {}
    data['image'] = []
    data['label'] = []
    f = open(data_txt_file, 'r')
    file_info = f.readlines()
    for i in file_info:
        img_info = i.strip('\n')
        name_label= img_info.split(",")
        img_name = name_label[0]
        img_label = name_label[1]
        img_label = int(img_label)
        img_read = io.imread(image_data_path+img_name+".jpg") #of dimension (8,8,3)
        #print(np.shape(img_read))
        data['image'].append(img_read)
        data['label'].append(img_label)
    data['image'] = np.array(data['image'])
    data['label'] = np.transpose([np.array(data['label'])])
    #print('image data shape:',data['image'])
    #print('label data shape:', np.shape(data['label']))
    #print(data['label'])
    #pass
    return data
