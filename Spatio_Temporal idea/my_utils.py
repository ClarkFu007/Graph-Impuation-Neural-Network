import csv
import random
import torch
import time
import numpy as np
import networkx as nx
from math import sqrt
from numpy import fabs, sum, square
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_dataset(datafile_B1, datafile_FA, datafile_FB, datafile_FC):
    """
    Inputs:
        the link of the data of Bus_1
        the link of the data of Feeder A
        the link of the data of Feeder B
        the link of the data of Feeder C
    Outputs:
        the complete dataset
    """
    Bus_1_X, Bus_1_y = input_data(datafile_B1, 3, 8760)
        # Data matrix Bus_1_X: (2, 8760)
        # Label matrix Bus_1_y: (1, 8760)
    Feeder_A_X, Feeder_A_y = input_data(datafile_FA, 51, 8760)
        # Data matrix Feeder_A_X: (34, 8760)
        # Label matrix Feeder_A_y: (17, 8760)
    Feeder_B_X, Feeder_B_y = input_data(datafile_FB, 180, 8760)
        # Data matrix Feeder_B_X: (120, 8760)
        # Label matrix Feeder_B_y: (60, 8760)
    Feeder_C_X, Feeder_C_y = input_data(datafile_FC, 486, 8760)
        # Data matrix Feeder_C_X (324, 8760)
        # Label matrix Feeder_C_X (162, 8760)

    # Get the dataset of shape (240, 3, 8760)
    data_set = np.zeros((240, 8760, 2), dtype='float')
    # data_set[0, 0:3, 0:8760] = np.vstack((Bus_1_X, Bus_1_y))
    data_set[0, 0:8760, 0:2] = np.transpose(Bus_1_X)
    data_set[1:18, 0:8760, 0] = Feeder_A_X[0:17, 0:8760]
    data_set[1:18, 0:8760, 1] = Feeder_A_X[17:34, 0:8760]
    # data_set[1:18, 2, 0:8760] = Feeder_A_y
    data_set[18:78, 0:8760, 0] = Feeder_B_X[0:60, 0:8760]
    data_set[18:78, 0:8760, 1] = Feeder_B_X[60:120, 0:8760]
    # data_set[18:78, 2, 0:8760] = Feeder_B_y
    data_set[78:240, 0:8760, 0] = Feeder_C_X[0:162, 0:8760]
    data_set[78:240, 0:8760, 1] = Feeder_C_X[162:324, 0:8760]
    # data_set[78:240, 2, 0:8760] = Feeder_C_y

    return data_set


def input_data(datafile_w, data_m, data_n):
    """
    Inputs:
        the location of the data file
        the number of row
        the number of column
    Outputs:
        data matrix X
        label matrix y
    """
    InputData = np.zeros((data_m, data_n), dtype='float')
    with open(datafile_w, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            data = [float(datum) for datum in row]
            InputData[i] = data
    bound_num = data_m * 2 // 3
    Data_X = InputData[0:bound_num, 0:data_n]  # Data matrix X
    Data_y = InputData[bound_num:data_m, 0:data_n]  # Label matrix y

    return Data_X, Data_y


def group_sample(data_set, time_size):
    """
    Inputs:
        the dataset of the 240-node test system
        the number of the time interval
    Outputs:
        the output window data with different sizes of time
    """
    node_num, total_num, fea_num = data_set.shape[:]

    sample_num = total_num // time_size
    OutputData = np.zeros((sample_num, node_num, time_size, fea_num), dtype='float')
    for sample_i in range(0, sample_num):
        sample_l = sample_i * time_size
        sample_u = sample_l + time_size
        OutputData[sample_i, 0:node_num, 0:time_size, 0:fea_num] = \
            data_set[0:node_num, sample_l:sample_u, 0:fea_num]

    """
    sample_num = total_num - time_size + 1
    OutputData = np.zeros((sample_num, node_num, fea_num, time_size), dtype='float')
    for sample_l in range(sample_num):
        sample_u = sample_l + time_size
        OutputData[sample_l, 0:node_num, 0:fea_num, 0:time_size] = \
            data_set[sample_l:sample_u, 0:node_num, 0:fea_num]
    """

    return OutputData


def myfunction():
    """
    Make the shuffle function behave the same
    """
    return 0.2


def split_sample(data_set, total_num, train_num, seed_value):
    """
    Inputs:
        the total dataset
        the total number of samples
        the number of train samples
    Output: the train set and the test set
    """
    index_list = list(range(total_num))
    # random.shuffle(index_list, myfunction)
    train_index = index_list[0: train_num]
    test_index = index_list[train_num: total_num]

    node_num, time_size, fea_num = data_set.shape[1:]

    train_set = np.zeros((train_num, node_num, time_size, fea_num), dtype='float')
    for sample_i in range(train_num):
        train_set[sample_i] = data_set[train_index[sample_i]]

    test_num = total_num - train_num
    test_set = np.zeros((test_num, node_num, time_size, fea_num), dtype='float')
    for sample_i in range(test_num):
        test_set[sample_i] = data_set[test_index[sample_i]]

    return train_set, test_set


def split_tr_te(dataset, time_step, split, seed):
    """
    Inputs:
        the total dataset
        the number of the time interval
        the threshold value to obtain the training and test sets
        the seed value to random sampling
    Output: the training set and the test set
    """
    dataset = group_sample(dataset, time_step)
    print("dataset", dataset.shape)
    sample_num = dataset.shape[0]
    # train_num = int(sample_num * split)
    train_num = 300
    train_set, test_set = split_sample(dataset, sample_num, train_num, seed)

    return train_set, test_set


def get_normalized_adj(A):
    """
    Input: the adjacency matrix
    Output: the degree normalized adjacency matrix
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def degrade_dataset(X, missingness, v, method, size):
    """
    Inputs:
        dataset to corrupt
        percentage of data to eliminate[0,1]
        replace with = 'zero' or 'np.nan'
        method = 'block' or 'nonblock'
        the block size
      Outputs:
        corrupted Dataset
        binary mask
    """
    if method == 'block':
        X_1d = X.flatten('F')  # According to the columns
        n = len(X_1d)
        mask_1d = np.ones(n)

        corrupt_ids = random.sample(range(n), int(missingness * n))
        for i in corrupt_ids:
            X_1d[i] = v
            mask_1d[i] = 0
            for j in range(size):
                if i - j >= 0:
                    X_1d[i - j] = v
                    mask_1d[i - j] = 0
                else:
                    X_1d[i + j] = v
                    mask_1d[i + j] = 0

        cX = X_1d.reshape(X.shape, order='F')
        mask = mask_1d.reshape(X.shape, order='F')
        return cX, mask

    elif method == 'nonblock':
        X_1d = X.flatten()
        n = len(X_1d)
        mask_1d = np.ones(n)

        corrupt_ids = random.sample(range(n), int(missingness * n))
        for i in corrupt_ids:
            X_1d[i] = v
            mask_1d[i] = 0

        cX = X_1d.reshape(X.shape)
        mask = mask_1d.reshape(X.shape)
        return cX, mask

    else:
        print("Pleast input the correct degrading method!")
        return False


if __name__ == "__main__":
    """
      The main function.
    """
    import torch.nn as nn
    m = nn.Dropout(p=0.5, inplace=True)
    input0 = torch.ones(5, 3)
    print(input0)
    output0 = m(input0)
    print(output0)

