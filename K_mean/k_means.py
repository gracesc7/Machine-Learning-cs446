from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset

# Make 3  clusters`
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
#data_file =
#with open('data/data/iris.data') as cvsfile:
read_df = pd.read_table('data/data/iris.data')
data_arr = read_df.as_matrix()
#print(np.shape(data_arr))
print(data_arr[0])
#print(np.shape(b))
#print(b[0])
print("Initial Centers")
print(C)
for i in range(np.shape(data_arr)[0]):
    data_arr[i][0] = data_arr[i][0].split(',')[:-1]   #remove the flower type name
    data_arr[i][0] = np.array(data_arr[i][0]).astype(float)
    #print(np.shape(data_arr[i][0]))
    #print(data_arr[i][0])
    #break
#data_arr = data_arr.astype(float)
#print(np.shape(data_arr))
#print(data_arr[0][0][0])
    #data_arr[i][0] = [float(k) for k in data_arr[i][0]]

#print("wa",type(data_arr[i][0]))
#print("len",len(data_arr[i][0]))#
#print(data_arr[i][0])
#print(data_arr[i][0][0])
#print(data_arr[0][0])
#iris = np.split(data_arr,3)    #split into equal sized three portions
#print('s',np.shape(iris))

#print(iris_3)
#print(b)
#print(read_df)
#print(read_df['V4'])
#data =
num_iter = 10
'''
def k_means(C):
    # Write your code here!
    for i in range(3):  #iterate through all types of flower
        temp_center = [[0]*4]*3
        for iter in range(num_iter):
            for j in range(np.shape(iris)[1]):  #iterate through every items in certain type
                distance = []
                for k in range(3):
                    #print("ir",len(iris[i][j]))
                    #print('ck',C[k])
                    d = list(map(lambda x,y:(x-y)**2,iris[i][j][0],C[k]))
                    distance.append(d)
                max_idx = distance.index(max(distance))
                #print('orig',temp_center[max_idx])
                #print(iris[i][j][0])
                temp_center[max_idx] = list(map(lambda x,y:(x+y),iris[i][j][0],temp_center[max_idx]))
                #print(temp_center[max_idx])
            for p in range(3):
                temp_center[p] = list(map(lambda x:x/50,temp_center[p]))
            print("temp",temp_center)
'''
def k_means(C):
    print('shape',np.shape(data_arr))
    temp_center = np.copy(C)
    #new_center = np.array(
    iter = 0
    while(1):
        print('iter',iter)
        center_sum = np.zeros(np.shape(C))
        count = [0]*3
        for j in range(np.shape(data_arr)[0]): #iterate through all 150    np.shape(data_arr)[0]
        #for j in range(3):
            #print('ir',data_arr[j][0])
            #print('orig',temp_center)
            eucli_dist = (data_arr[j][0] - temp_center)**2
            #print('eucli',eucli_dist)
            eucli_dist = np.sum(eucli_dist,axis = 1)
            #print('sum',eucli_dist)
            min_idx = np.argmin(eucli_dist,axis = 0)
            #print('max_idx',max_idx)
            center_sum[min_idx] = center_sum[min_idx] + data_arr[j][0]
            #print('new center',center_sum)
            count[min_idx] += 1
        prev_center = np.copy(temp_center)
        for p in range(3):
            #print('c',count[p])
            #print('p',center_sum[p])
            if(count[p] == 0):
                temp_center[p] = np.array([0]*4)
            else:
                temp_center[p] = center_sum[p]/count[p]
        print('center',temp_center)
        abs_err = np.sum(np.fabs(temp_center - prev_center))
        print(abs_err)
        iter += 1
        if(abs_err <= 1e-3):
            break
    return temp_center
