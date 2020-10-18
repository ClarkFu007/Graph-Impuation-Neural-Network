import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC


class TimeBlock(nn.Module, ABC):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, kernel_size),
                               padding=(0, 1), padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, kernel_size),
                               padding=(0, 1), padding_mode='reflect')
        self.conv3 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, kernel_size),
                               padding=(0, 1), padding_mode='reflect')

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes,
                                       num_timesteps,
                                       num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
                                       num_timesteps_out,
                                       num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)

        return out


class STGCNBlock(nn.Module, ABC):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """

        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.leaky_relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class stgcn_autoencoder(nn.Module, ABC):
    """
    Autoencoder with spatio-temporal graph convolutional layers.
    """
    def __init__(self, num_nodes, in_feat, size_step, time_step):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps: Number of time steps fed into the network.
        :param size_step: Adjustable dimension of features at each layer
        """
        super(stgcn_autoencoder, self).__init__()
        self.time_step = time_step
        out_feat = in_feat

        self.drop_out = nn.Dropout(p=0.4)

        self.block1 = STGCNBlock(in_channels=in_feat, spatial_channels=in_feat+1*size_step,
                                 out_channels=in_feat+4*size_step, num_nodes=num_nodes)
        # self.linear_nn1 = nn.Linear(in_feat+4*size_step, in_feat+4*size_step)
        self.block2 = STGCNBlock(in_channels=in_feat+4*size_step, spatial_channels=in_feat+1*size_step,
                                 out_channels=in_feat, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=in_feat, out_channels=in_feat)
        self.linear_nn2 = nn.Linear(in_feat, out_feat)

    def forward(self, A_hat, X):
        """
        :param A_hat: Normalized adjacency matrix.
        :param X: Input data of shape (batch_size, num_nodes,
                                       num_timesteps, num_features).
        """
        h1 = X
        # h1 = self.drop_out(X)
        h1 = self.block1(h1, A_hat)
        h2 = self.block2(h1, A_hat)
        # h2 = self.linear_nn1(h2)
        h3 = self.last_temporal(h2)
        h4 = self.linear_nn2(h3)
        h4 = torch.sigmoid(h4)

        return h4



