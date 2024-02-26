import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        batch_size, in_channels, input_height, input_width = A.shape
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1

        self.Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

        for i in range(batch_size):
            for j in range(self.out_channels):
                for k in range(in_channels):
                    for y in range(output_height):
                        for x in range(output_width):
                        # Perform element-wise multiplication and sum the results
                            self.Z[i, j, y, x] += np.sum(A[i, k, y:y+self.kernel_size, x:x+self.kernel_size] * self.W[j, k, :, :])
                # Add bias after summing over all input channels
                self.Z[i, j, :, :] += self.b[j]

        return self.Z
        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        
        # Pad the dLdZ map with K −1 zeros.
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)), 'constant', constant_values=0)
        # Flip the filter top to bottom and left to right.
        W_flipped = np.flip(self.W, (2, 3))
        # Perform a convolution between the padded dLdZ and the flipped filter.
        dLdA = np.zeros(self.A.shape)
        batch_size, in_channels, input_height, input_width = self.A.shape

        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(self.out_channels):
                    for y in range(input_height):
                        for x in range(input_width):
                            dLdA[i, j, y, x] += np.sum(dLdZ_padded[i, k, y:y+self.kernel_size, x:x+self.kernel_size] * W_flipped[k, j, :, :])


        
        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

        output_height, output_width = dLdZ.shape[2], dLdZ.shape[3]
        # dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        # A (np.array): (batch_size, in_channels, input_height, input_width)
        # dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        # convolve dLdZ with A to get dLdW
         
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                for k in range(batch_size):
                    for y in range(input_height - output_height + 1):
                        for x in range(input_width - output_width + 1):
                            self.dLdW[i, j, y, x] += np.sum(self.A[k, j, y:y+output_height, x:x+output_width] * dLdZ[k, i, :, :])  
                        
               
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        return dLdA 


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant', constant_values=0)

        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample2d.forward(Z)    

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Unpad the gradient
        dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dLdA
