import copy
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import fabs, sum, square
from math import sqrt
from sklearn import preprocessing
from MIDA_utils import corrupt_dataset, merge_feature


class Autoencoder(nn.Module):
    """
       The proposed imputation model.
    """
    def __init__(self, dim, theta):
        super(Autoencoder, self).__init__()
        self.dim = dim
        self.drop_out = nn.Dropout(p=0.5)

        self.encoder = nn.Sequential(
            nn.Linear(dim + theta * 0, dim + theta * 1),
            nn.Tanh(),
            nn.Linear(dim + theta * 1, dim + theta * 2),
            nn.Tanh(),
            nn.Linear(dim + theta * 2, dim + theta * 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim + theta * 3, dim + theta * 2),
            nn.Tanh(),
            nn.Linear(dim + theta * 2, dim + theta * 1),
            nn.Tanh(),
            nn.Linear(dim + theta * 1, dim + theta * 0)
        )

    def forward(self, x):
        x = x.view(-1, self.dim)
        # x_missed = self.drop_out(x)
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, self.dim)

        return out


def do_MIDA(train_set, validation_set, test_set, missingness,
            verbose, corrupt_method, block_size, iter_num, theta, epochs):
    """
       The core function to do MIDA imputation.
    :param train_set: training set
    :param validation_set: validation set
    :param test_set: test set
    :param missingness: value of missingness
    :param verbose: whther to print the training process
    :param corrupt_method: "block" or "nonblock"
    :param block_size: size of block
    :param iter_num: number of imputations
    :param theta: increased dimension for the autoencoder
    :param epochs: number of epochs for training

    :output: the imputation performance
    """
    # Degrade three sets
    x_train = copy.deepcopy(train_set)  # The feature matrix of train
    x_validation = copy.deepcopy(validation_set)  # The feature matrix of train
    x_test = copy.deepcopy(test_set)  # The feature matrix of test

    v_value = 0  # You can replace the missing value as 0 or np.nan
    cx_train, cx_train_mask = corrupt_dataset(x_set=x_train,
                                              missingness=missingness,
                                              v_value=v_value,
                                              corrupt_method=corrupt_method,
                                              block_size=block_size)
    cx_validation, cx_validation_mask = corrupt_dataset(x_set=x_validation,
                                                        missingness=missingness,
                                                        v_value=v_value,
                                                        corrupt_method=corrupt_method,
                                                        block_size=block_size)
    cx_test, cx_test_mask = corrupt_dataset(x_set=x_test,
                                            missingness=missingness,
                                            v_value=v_value,
                                            corrupt_method=corrupt_method,
                                            block_size=block_size)
    # After merging features
    num_cols = [0, 1]
    # For training data
    x_train = merge_feature(x_train, num_cols)
    cx_train = merge_feature(cx_train, num_cols)
    cx_train_mask = merge_feature(cx_train_mask, num_cols)
    # print("After merging features, x_train", x_train.shape)
    # print("After merging features, cx_train", cx_train.shape)
    # print("After merging features, cx_train_mask", cx_train_mask.shape)
    # For validation data
    x_validation = merge_feature(x_validation, num_cols)
    cx_validation = merge_feature(cx_validation, num_cols)
    cx_validation_mask = merge_feature(cx_validation_mask, num_cols)
    # print("After merging features, x_validation", x_validation.shape)
    # print("After merging features, cx_validation", cx_validation.shape)
    # print("After merging features, cx_validation_mask", cx_validation_mask.shape)
    # For test data
    x_test = merge_feature(x_test, num_cols)
    cx_test = merge_feature(cx_test, num_cols)
    cx_test_mask = merge_feature(cx_test_mask, num_cols)
    # print("After merging features, x_test", x_test.shape)
    # print("After merging features, cx_test", cx_test.shape)
    # print("After merging features, cx_test_mask", cx_test_mask.shape)

    from MIDA_utils import preprocess_data
    x_train0, cx_train0 = preprocess_data(x_train, cx_train)
    x_validation0, cx_validation0 = preprocess_data(x_validation, cx_validation)  # Pass by reference!
    x_test0, cx_test0 = preprocess_data(x_test, cx_test)

    # print("After preprocessing, x_train0, cx_train0", x_train0.shape, cx_train0.shape)
    # print("After preprocessing, x_validation0, cx_validation0", x_validation0.shape, cx_validation0.shape)
    # print("After preprocessing, x_test0, cx_test0", x_test0.shape, cx_test0.shape)

    # Define the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from MIDA_core import Autoencoder
    model = Autoencoder(dim=x_train0.shape[2], theta=theta).to(device)
    # Define Loss and Optimizer
    loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.99, lr=0.01, nesterov=True)

    # Train the model
    sample_num = x_train0.shape[0]
    dur_train = np.zeros((epochs, sample_num), dtype='float')
    x_trainT = torch.from_numpy(x_train0).float()
    cx_trainT = torch.from_numpy(cx_train0).float()
    train_loader = torch.utils.data.DataLoader(dataset=cx_trainT,
                                               batch_size=1,
                                               shuffle=True)
    for iter_i in range(1, iter_num + 1):
        minimum_MA_error = 10  # Make it big enough
        for epoch in range(epochs):
            t0 = time.time()
            model.train()
            for i, batch_data in enumerate(train_loader):
                batch_data = batch_data.squeeze()
                batch_data = batch_data.to(device)
                reconst_data = model(batch_data)
                cost = loss(reconst_data, x_trainT[i])
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                dur_t0 = time.time() - t0
                dur_train[epoch, i] = dur_t0

            # For validation
            vali_real_mean, vali_MA_error, vali_RME_error, \
            vali_nAE, vali_nAAE = get_results(cx_set0=cx_validation0, cx_mask=cx_validation_mask,
                                              device=device, imputation_model=model, cx_set=cx_validation,
                                              x_set=x_validation)
            if verbose:
                print('Epoch [%d/%d], for valadation data, '
                      'real mean value: %.6f, '
                      'mean absolute error: %.6f, '
                      'root mean squared error: %.6f, '
                      'normalized absolute error: %.6f, '
                      'normalized accumulate absolute error: %.6f'
                      % (epoch + 1, epochs, vali_real_mean, vali_MA_error,
                         vali_RME_error, vali_nAE, vali_nAAE))
            if minimum_MA_error > vali_MA_error:
                minimum_MA_error = vali_MA_error
                torch.save({"auto_state_dict": model.state_dict(),
                            "optim_auto_state_dict": model.state_dict()},
                           "best_model_" + str(iter_i) + ".pth")

    # Test the model
    test_real_mean, test_MA_error = 0, 0
    test_RME_error = 0
    test_nAE, test_nAAE = 0, 0
    for iter_i in range(1, iter_num + 1):
        model_path = "best_model_" + str(iter_i) + ".pth"
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["auto_state_dict"])
        model.load_state_dict(checkpoint["optim_auto_state_dict"])
        temp_test_real_mean, temp_test_MA_error, temp_test_RME_error, \
        temp_test_nAE, temp_test_nAAE = get_results(cx_set0=cx_test0, cx_mask=cx_test_mask,
                                                    device=device, imputation_model=model,
                                                    cx_set=cx_test, x_set=x_test)
        test_real_mean += temp_test_real_mean
        test_MA_error += temp_test_MA_error
        test_RME_error += temp_test_RME_error
        test_nAE += temp_test_nAE
        test_nAAE += temp_test_nAAE

    print('    For test data when missingness is %.2f, block size is %d' % (missingness, block_size))
    print('real mean value: %.6f' % (test_real_mean / iter_num))
    print('mean absolute error: %.6f' % (test_MA_error / iter_num))
    print('root mean squared error: %.6f' % (test_RME_error / iter_num))
    print('normalized absolute error: %.6f' % (test_nAE / iter_num))
    print('normalized accumulate absolute error: %.6f' % (test_nAAE / iter_num))
    print(' ')


def do_imputation(sample_i, cx_set0, cx_mask, device, imputation_model, cx_set, x_set):
    """
       Do the imputation with the trained model (sub-function to get final results).
    :param sample_i: the ith sample
    :param cx_set0: corrupted data after preprocessing
    :param cx_mask: mask of corrupted data
    :param device: the running device
    :param imputation_model: the trained model
    :param cx_set: corrupted data before preprocessing
    :param x_set: real data before preprocessing

    :return: miss_num, x_real, error
    """
    missed_data = cx_set0[sample_i]
    mask = cx_mask[sample_i]
    missed_dataT = np.multiply(missed_data, mask)
    missed_dataT = torch.FloatTensor(missed_dataT).to(device)

    # Reconstruction
    with torch.no_grad():
        imputation_model.eval()
        filled_data = imputation_model(missed_dataT)
    filled_data = filled_data.cpu().detach().numpy()

    max_vector = np.expand_dims(cx_set[sample_i].max(axis=0), axis=0)
    filled_data = np.multiply(filled_data, max_vector)
    """
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(cx_set[sample_i])
    filled_data = scaler.inverse_transform(filled_data)
    """

    total_num = cx_set0.shape[1] * cx_set0.shape[2]
    miss_num = total_num - np.count_nonzero(cx_mask[sample_i])
    new_mask = np.ones((cx_mask.shape[1], cx_mask.shape[2]))
    new_mask = new_mask - cx_mask[sample_i]

    imputed_data = np.multiply(filled_data, new_mask)
    x_real = np.multiply(x_set[sample_i], new_mask)

    error = imputed_data - x_real

    return miss_num, x_real, error


def get_results(cx_set0, cx_mask, device,
                imputation_model, cx_set, x_set):
    """
       Get the performance of imputation (main function to get final results).
    :param cx_set0: corrupted data after preprocessing
    :param cx_mask: mask of corrupted data
    :param device: the running device
    :param imputation_model: the trained model
    :param cx_set: corrupted data before preprocessing
    :param x_set: real data before preprocessing

    :return: real_mean, MA_error, RME_error, nAE, nAAE
    """
    MA_error, RME_error = 0, 0
    nAE, nAAE = 0, 0
    real_mean = 0
    total_miss = 0
    for sample_i in range(cx_set0.shape[0]):
        miss_num, x_real, error = do_imputation(sample_i=sample_i, cx_set0=cx_set0,
                                                cx_mask=cx_mask, device=device,
                                                imputation_model=imputation_model,
                                                cx_set=cx_set, x_set=x_set)
        mean_sample_i = np.mean(x_set[sample_i])
        total_miss += miss_num
        real_mean += sum(fabs(x_real))
        MA_error += sum(fabs(error))
        RME_error += sum(square(error))
        nAE += sum(fabs(error)) / mean_sample_i
        nAAE += fabs(sum(error)) / mean_sample_i

    return real_mean / total_miss, MA_error / total_miss, \
           sqrt(RME_error / total_miss), nAE / total_miss, nAAE / total_miss