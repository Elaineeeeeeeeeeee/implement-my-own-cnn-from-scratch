# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import sys
sys.path.append('mytorch')
sys.path.append('mytorch/nn')
from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        # inchannel, outchannel, kernel_size, stride   
        self.conv1 = Conv1d(24, 8, 8, 4)
        self.conv2 = Conv1d(8, 16, 1, 1)
        self.conv3 = Conv1d(16, 4, 1, 1)

        self.layers = [ self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]


    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights

        # load_W: out channels, kernel size, in channels
        # W: out_channels, in_channels, kernel_size
        def load_weight(conv, w):
            in_channels = conv.conv1d_stride1.in_channels
            out_channels = conv.conv1d_stride1.out_channels
            kernel_size = conv.conv1d_stride1.kernel_size

            assert w.shape[1] == out_channels

            w = w.reshape((kernel_size, in_channels, out_channels))
            w = w.transpose((2, 1, 0))
            conv.conv1d_stride1.W = w
        load_weight(self.conv1, w1)
        load_weight(self.conv2, w2)
        load_weight(self.conv3, w3)

       

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        print(A.shape)
        # 1,24,128
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        # inchannel, outchannel, kernel_size, stride

        self.conv1 = Conv1d(24, 8, 8 , 4)
        self.conv2 = Conv1d(8, 16, 8, 4)
        self.conv3 = Conv1d(16, 4, 8, 4)

        self.layers = [ self.conv1, ReLU(), self.conv2, ReLU(), self.conv3]


    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        self.conv1.conv1d_stride1.W = w1.reshape(8, 24, 8)
        self.conv2.conv1d_stride1.W = w2.reshape(16, 8, 1)
        self.conv3.conv1d_stride1.W = w3.reshape(4, 16, 1).transpose(2, 0, 1)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
