import csv
import random
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def do_GINN(train_set, test_set, time_step, missingness,
            miss_value, miss_method, block_size,
            theta, epoch_num, size_step, auto_lr, fine_tune):
    """
    Inputs:
        the taining set
        the test set
        the number of the time interval
        the percentile of the missingness
        the value for the missing postion: 0 or np.nan
        the method to degrading the dataset 'block' or 'nonblock'
        the size for the 'block' method
        the theta value for the cost function
        the number of epochs while training
        the value of the learning rate
        the step size to increase the dimension of the autoencoder at each layer
        whether to fine tune or not
    Outputs:
        the complete dataset
    """
    num_cols = [0, 1]
    cat_cols = [2]

    import copy
    x_train = copy.deepcopy(train_set)  # The feature matrix of train
    x_test = copy.deepcopy(test_set)  # The feature matrix of test
    from my_utils import degrade_dataset
    cx_train = np.zeros((x_train.shape[0], x_train.shape[1],
                         x_train.shape[2], x_train.shape[3]), dtype='float')
    cx_train_mask = np.zeros((x_train.shape[0], x_train.shape[1],
                              x_train.shape[2], x_train.shape[3]), dtype='int')
    for sample_i in range(x_train.shape[0]):
        for node_i in range(x_train.shape[1]):
            cx_train[sample_i, node_i], cx_train_mask[sample_i, node_i] = \
                degrade_dataset(X=x_train[sample_i, node_i], missingness=missingness,
                                v=miss_value, method=miss_method, size=block_size)

    cx_test = np.zeros((x_test.shape[0], x_test.shape[1],
                        x_test.shape[2], x_test.shape[3]), dtype='float')
    cx_test_mask = np.zeros((x_test.shape[0], x_test.shape[1],
                             x_test.shape[2], x_test.shape[3]), dtype='int')
    for sample_i in range(x_test.shape[0]):
        for node_i in range(x_test.shape[1]):
            cx_test[sample_i, node_i], cx_test_mask[sample_i, node_i] = \
                degrade_dataset(X=x_test[sample_i, node_i], missingness=missingness,
                                v=miss_value, method=miss_method, size=block_size)

    from sklearn import preprocessing
    # For the train part:
    oh_real_tr = copy.deepcopy(x_train)
    oh_x_tr = copy.deepcopy(cx_train)
    for sample_i in range(cx_train.shape[0]):
        for node_i in range(cx_train.shape[1]):
            scaler_tr1 = preprocessing.MinMaxScaler()
            oh_x_tr[sample_i, node_i] = \
                scaler_tr1.fit_transform(oh_x_tr[sample_i, node_i])

            scaler_tr2 = preprocessing.MinMaxScaler()
            oh_real_tr[sample_i, node_i] = \
                scaler_tr2.fit_transform(oh_real_tr[sample_i, node_i])

    print("oh_x_tr", oh_x_tr.shape)
    print("oh_real_tr", oh_real_tr.shape)

    # For the test part:
    oh_x_te = copy.deepcopy(cx_test)
    for sample_i in range(cx_test.shape[0]):
        for node_i in range(cx_test.shape[1]):
            scaler_te = preprocessing.MinMaxScaler()
            oh_x_te[sample_i, node_i] = \
                scaler_te.fit_transform(oh_x_te[sample_i, node_i])

    print("oh_x_te", oh_x_te.shape)

    from my_core import GINN
    import torch.nn.functional as F
    my_model = GINN(oh_x_tr=oh_x_tr, oh_real_tr=oh_real_tr,
                    oh_mask_tr=cx_train_mask, theta=theta,
                    time_step=time_step, size_step=size_step,
                    num_cols=num_cols, cat_cols=cat_cols, auto_lr=auto_lr)

    my_model.fit(epochs=epoch_num, fine_tune=fine_tune)
    my_model.transform(train_set=train_set, test_set=test_set,
                       cx_train=cx_train, cx_test=cx_test, oh_x_te=oh_x_te,
                       cx_train_mask=cx_train_mask, cx_test_mask=cx_test_mask)








if __name__ == "__main__":
    """
      The main function.
    """
    a = [1, 2, 3]
    b = a
    c = a[:]
    a[0] = 2
    print(b)
    print(c)