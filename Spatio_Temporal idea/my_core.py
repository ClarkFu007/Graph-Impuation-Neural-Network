import dgl
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from numpy import fabs, sum, square
from my_utils import *
from my_models import *

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class GINN(object):
    """
    Impute with GINN!
    Functions:
        __init__
        fit()
        transform()
    """

    def __init__(self, oh_x_tr, oh_real_tr, oh_mask_tr, theta,
                 time_step, size_step, num_cols, cat_cols,
                 auto_lr=0.001, weight_decay=0):
        """
            Build the graph-structure of the dataset based on the file
            Instantiate the network based on the graph using the dgl library
        """
        self.sample_num = oh_x_tr.shape[0]  # The number of samples
        self.num_nodes = oh_x_tr.shape[1]  # The number of nodes of each graph
        self.feat_num = oh_x_tr.shape[3]  # The number of features
        self.oh_x_tr = oh_x_tr  # The corrupted training set
        self.oh_real_tr = oh_real_tr  # The real training set
        self.oh_mask_tr = oh_mask_tr  # The mask of the corrupted training set

        self.theta = theta  # the theta value for the cost function
        self.time_step = time_step  # The time step of the dataset
        self.size_step = size_step  # the step size to increase the dimension of the autoencoder

        self.num_cols = num_cols  # numerical_columns
        self.cat_cols = cat_cols  # categorical_columns
        self.auto_lr = auto_lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("GINN is running on", self.device)

        """
           Construct the graph.
        """
        datafile_w = "Edge info.csv"
        with open(datafile_w, 'r') as f:  # Count how many edges
            reader = csv.reader(f)
            row_num = 0
            for row in reader:
                row_num += 1
            graph_edges = np.zeros((row_num, 2), dtype='int')
        with open(datafile_w, 'r') as f:  # Report the edge pairs
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                data = [int(datum) for datum in row]
                graph_edges[i] = data

        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for row in range(graph_edges.shape[0]):  # Add edges to the graph
            self.adj_matrix[graph_edges[row, 0], graph_edges[row, 1]] = 1
            self.adj_matrix[graph_edges[row, 1], graph_edges[row, 0]] = 1
        print("The adjacency_matrix is", self.adj_matrix.shape)

        from my_utils import get_normalized_adj
        adj_wave = get_normalized_adj(self.adj_matrix)  # The degree normalized adjacency matrix.
        self.adj_wave = torch.FloatTensor(adj_wave).to(self.device)

        from my_models import stgcn_autoencoder
        """
        Autoencoder with spatio-temporal graph convolutional layers.
        """
        # construct an autoencoder
        self.autoencoder = stgcn_autoencoder(num_nodes=self.num_nodes,
                                             in_feat=self.feat_num,
                                             size_step=self.size_step,
                                             time_step=self.time_step).to(self.device)
        # construct an optimizer
        self.optim_auto = torch.optim.Adam(self.autoencoder.parameters(),
                                           lr=self.auto_lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=weight_decay)
        # and a learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim_auto,
                                                            step_size=5,
                                                            gamma=0.8)
        return

    def fit(self, epochs, fine_tune, verbose=True):
        """
            Trains the network, if fine_tune=True uses the previous state
            of the optimizer instantiated before.
        """
        if fine_tune:
            checkpoint = torch.load("ginn.pth")
            self.autoencoder.load_state_dict(checkpoint["auto_state_dict"])
            self.optim_auto.load_state_dict(checkpoint["optim_auto_state_dict"])

        """
           Create criterions with respect to categorical columns and
        numerical columns.
        """
        if self.theta > 0.0:
            mse_criterion = nn.MSELoss().to(self.device)
            global_criterion = nn.MSELoss().to(self.device)
        if self.theta < 1.0:
            bce_criterion = nn.BCELoss().to(self.device)

        start0 = time.time()
        batch_size = 30
        early_stop = False
        for epoch in range(epochs):
            if early_stop:
                break

            t0, total_loss = time.time(), 0
            permutation = torch.randperm(self.sample_num)
            for batch_i in range(0, self.sample_num, batch_size):
                self.autoencoder.train()
                self.optim_auto.zero_grad()  # Zero the gradients

                indices = permutation[batch_i:batch_i + batch_size]

                featT = self.oh_x_tr
                featT_r = self.oh_real_tr
                featT = torch.FloatTensor(featT)
                featT_r = torch.FloatTensor(featT_r)

                X_batch, y_batch = featT[indices], featT_r[indices]
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                iX = self.autoencoder(self.adj_wave, X_batch)  # Reconstruction
                num_loss = mse_criterion(iX, y_batch)
                           # global_criterion(torch.mean(y_batch, dim=0), torch.mean(iX, dim=0))
                num_loss.backward()  # Calculate the gradients
                self.optim_auto.step()  # Update the weights

                total_loss += num_loss.item()

            self.lr_scheduler.step()  # update the learning rate
            mean_loss = total_loss
            if verbose:
                print("Epoch is %04d/%04d, mean loss is: %f," % (epoch + 1, epochs, mean_loss))
            if mean_loss < 1e-05:
                early_stop = True

            torch.save(
                {
                    "auto_state_dict": self.autoencoder.state_dict(),
                    "optim_auto_state_dict": self.optim_auto.state_dict()
                },
                "ginn.pth"
            )

        end0 = time.time()
        total_time = end0 - start0
        print("The total running time is %f seconds" % (total_time))
        if epochs != 0:
            mean_time = total_time / epochs
            print("The mean training time is %f seconds" % (mean_time))
        # print("The mean training time is %f seconds" % (dur_train.mean()))

        return

    def transform(self, train_set, test_set, cx_train, cx_test,
                  oh_x_te, cx_train_mask, cx_test_mask):
        """
           Test the trained model.
        """
        # For the training part
        features = self.oh_x_tr
        featT = torch.from_numpy(features).float()
        featT = featT.to(self.device)
        # Reconstruction
        with torch.no_grad():
            self.autoencoder.eval()
            imputed_data = self.autoencoder(self.adj_wave, featT)
        iX = imputed_data.detach().cpu().numpy()

        Train_MA_error, Train_RME_error = 0, 0
        total_miss = 0
        for sample_i in range(cx_train.shape[0]):
            for node_i in range(cx_train.shape[1]):
                max_vector = np.expand_dims(cx_train[sample_i, node_i].max(axis=0), axis=0)
                imputed_tr = np.multiply(iX[sample_i, node_i], max_vector)
                """
                scaler_tr = preprocessing.MinMaxScaler()
                scaler_tr.fit(cx_train[sample_i, node_i])
                imputed_tr = scaler_tr.inverse_transform(iX[sample_i, node_i])
                """
                total_num = cx_train.shape[2] * cx_train.shape[3]
                miss_num = total_num - np.count_nonzero(cx_train_mask[sample_i, node_i])
                new_mask = np.ones((cx_train_mask.shape[2], cx_train_mask.shape[3]))
                new_mask = new_mask - cx_train_mask[sample_i, node_i]

                imputed_tr = np.multiply(imputed_tr, new_mask)
                train_r = np.multiply(train_set[sample_i, node_i], new_mask)

                error = fabs(train_r - imputed_tr)

                total_miss += miss_num
                Train_MA_error += sum(error)
                Train_RME_error += sum(square(error))

        print("For the training set, the mean absolute error is %f" % (Train_MA_error / total_miss))
        print("For the training set, the root mean squared error is %f" % (sqrt(Train_RME_error / total_miss)))

        # For the test part
        dur_impu = np.zeros(test_set.shape[0], dtype='float')
        features = oh_x_te
        featT = torch.from_numpy(features).float()
        featT = featT.to(self.device)

        # Reconstruction
        self.autoencoder.eval()
        with torch.no_grad():
            imputed_data = self.autoencoder(self.adj_wave, featT)
        iX = imputed_data.detach().cpu().numpy()

        Test_MA_error, Test_RME_error = 0, 0
        total_miss = 0
        for sample_i in range(cx_test.shape[0]):
            for node_i in range(cx_test.shape[1]):
                max_vector = np.expand_dims(cx_test[sample_i, node_i].max(axis=0), axis=0)
                imputed_te = np.multiply(iX[sample_i, node_i], max_vector)
                """
                scaler_te = preprocessing.MinMaxScaler()
                scaler_te.fit(cx_test[sample_i, node_i])
                imputed_te = scaler_te.inverse_transform(iX[sample_i, node_i])
                """
                total_num = cx_test.shape[2] * cx_test.shape[3]
                miss_num = total_num - np.count_nonzero(cx_test_mask[sample_i, node_i])
                new_mask = np.ones((cx_test_mask.shape[2], cx_test_mask.shape[3]))
                new_mask = new_mask - cx_test_mask[sample_i, node_i]

                imputed_te = np.multiply(imputed_te, new_mask)
                test_r = np.multiply(test_set[sample_i, node_i], new_mask)
                error = fabs(test_r - imputed_te)

                total_miss += miss_num
                Test_MA_error += sum(error)
                Test_RME_error += sum(square(error))

        print("For the test set, the mean absolute error is %f" % (Test_MA_error / total_miss))
        print("For the test set, the root mean squared error is %f" % (sqrt(Test_RME_error / total_miss)))
        print("The mean imputation time is %f seconds" % (dur_impu.mean()))
