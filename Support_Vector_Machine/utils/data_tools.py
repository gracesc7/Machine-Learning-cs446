"""Implements feature extraction and other data processing helpers.
(This file will not be graded).
"""

import numpy as np
import skimage
from skimage import color


def preprocess_data(data, process_method='default'):
    """Preprocesses dataset.

    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1] by dividing by 255.
          2. Remove dataset mean. Average the images across the batch dimension.
             This will result in a mean image of dimension (8,8,3).
          3. Flatten images, data['image'] is converted to dimension (N, 8*8*3)
        if process_method is 'default':
          1. Convert images to range [0,1]
          2. Convert from rgb to gray then back to rgb. Use skimage
          3. Take the absolute value of the difference with the original image.
          4. Remove dataset mean. Average the absolute value differences across
             the batch dimension. This will result in a mean of dimension (8,8,3).
          5. Flatten images, data['image'] is converted to dimension (N, 8*8*3)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """

    N = np.shape(data['image'])[0]
    if process_method == 'raw':
        img_sum = np.zeros((8,8,3))
        data['image'] = data['image'] /255
        for img in data['image']:
            img_sum = np.add(img,img_sum)
        img_mean = np.divide(img_sum , N)
        #print("img_mean:", np.shape(img_mean))
        data_update = []
        for img in data['image']:
            img_update = img - img_mean
            data_update.append(img_update)
        data_update = np.array(data_update)
        data['image'] = np.reshape(data_update,(N,8*8*3))
        #print("data_update0:",data_update[0])
        #print("data_update1:",data_update[1])
        #print(np.shape(data['image']))
        #print(data_update)
        #np.reshape(data_update)
        #pass

    elif process_method == 'default':
        img_sum = np.zeros((8,8,3))
        original = data['image'] /255
        #print("original0:",data['image'][0])
        #print("original1:",data['image'][1])
        data['image'] = skimage.color.rgb2gray(data['image'])
        update = skimage.color.gray2rgb(data['image'])
        diff = update -original
        diff = np.absolute(diff)
        for img in diff:
            img_sum = np.add(img,img_sum)
        img_mean = np.divide(img_sum , N)
        #print("img_Mean:",img_mean)
        data_update = []
        for img in diff:
            diff_update = img - img_mean
            data_update.append(diff_update)
        data_update = np.array(data_update)
        data['image'] = np.reshape(data_update,(N,8*8*3))
        #print("finalshape:",np.shape(data['image']))
        #print(data['image'])
        #print("update0:",data['image'][0])
        #print("update1:",data['image'][1])
        #print(np.shape(data['image']))
        #pass

    elif process_method == 'custom':
        # Design your own feature!
        pass
    return data


def compute_image_mean(data):
    """ Computes mean image.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        image_mean(numpy.ndarray): Avaerage across the example dimension.
    """
    image_mean = None
    pass
    return image_mean


def remove_data_mean(data):
    """Removes data mean.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    pass
    return data
