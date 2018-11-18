"""Input and output helpers to load in data.
"""
import numpy as np

def read_dataset_tf(path_to_dataset_folder,index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1],
                                                     [1, x2],
                                                     [1, x3],
                                                     .......]
                                where xi is the 16-dimensional feature of each sample

        T(numpy.ndarray): class label vector T = [[y1],
                                                  [y2],
                                                  [y3],
                                                   ...]
                             where yi is 1/0, the label of each sample
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    f = open(path_to_dataset_folder + '/' + index_filename, 'r')
    features = f.readlines()
    T = []
    A = []
    for i in features:
        txtpath = i.strip('\n')
        if(txtpath[0] == '-'):
            txtpath = txtpath[3:]
            label = int(0)
        else:
            txtpath = txtpath[2:]
            label = int(i[0])
        T.append(label)
        txtfile = open(path_to_dataset_folder + '/' + txtpath, 'r')
        x = txtfile.readline().strip('\r\n')
        x_arr = x[3:].split()
        x_arr = [float(i) for i in x_arr]
        x_arr = [1]+x_arr
        A.append(x_arr)
    A = np.array(A)
    T = np.transpose([np.array(T)])
    #print("T:", T)
    #print("T:", np.shape(T) )
    #print("A", np.shape(A))
    return A,T
