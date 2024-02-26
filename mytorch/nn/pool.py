import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, out_channels = A.shape[0], A.shape[1]
        output_width, output_height = A.shape[2] - self.kernel + 1, A.shape[3] - self.kernel + 1 
        Z = np.zeros((batch_size, out_channels, output_width, output_height))

        # Initialize cache to store the indices
        self.cache = np.zeros((batch_size, out_channels, output_width, output_height, 2), dtype=int)

        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(output_width):
                    for l in range(output_height):
                        # Extract the current window
                        window = A[i, j, k:k + self.kernel, l:l + self.kernel]
                
                        # Find the max value in the current window
                        Z[i, j, k, l] = np.max(window)
                
                        # Find the index of the max value within the window
                        max_index = np.unravel_index(np.argmax(window), window.shape)
                
                        # Store the multi-dimensional index
                        self.cache[i, j, k, l] = (k + max_index[0], l + max_index[1])      
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2] + self.kernel - 1, dLdZ.shape[3] + self.kernel - 1))
        
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for k in range(dLdZ.shape[2]):
                    for l in range(dLdZ.shape[3]):
                        # Find the index of the max value during forward pass
                        max_index = self.cache[i, j, k, l]
                        # Convert the index to 2D coordinates
                        max_y = max_index[0]
                        max_x = max_index[1]
                        # Propagate the gradient to the max location
                        dLdA[i, j, max_y, max_x] += dLdZ[i, j, k, l]


        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, out_channels = A.shape[0], A.shape[1]
        output_width, output_height = A.shape[2] - self.kernel + 1, A.shape[3] - self.kernel + 1 
        Z = np.zeros((batch_size, out_channels, output_width, output_height))
        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(output_width):
                    for l in range(output_height):
                        Z[i, j, k, l] = np.mean(A[i, j, k:k + self.kernel, l:l + self.kernel])

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2] + self.kernel - 1, dLdZ.shape[3] + self.kernel - 1))
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for k in range(dLdZ.shape[2]):
                    for l in range(dLdZ.shape[3]):
                        # Get the gradient for the current position
                        grad = dLdZ[i, j, k, l]
                        for m in range(self.kernel):
                            for n in range(self.kernel):
                                # Distribute the gradient to the corresponding window in dLdA
                                dLdA[i, j, k+m, l+n] += grad / (self.kernel * self.kernel)
                
                        
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)  

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
    
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)
        return dLdA
