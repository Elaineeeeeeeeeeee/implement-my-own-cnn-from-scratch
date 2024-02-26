import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        #insert self.upsampling_factor - 1 zeros between each element of A
        Z = np.zeros((A.shape[0], A.shape[1], (A.shape[2]-1) * self.upsampling_factor + 1))
        Z[:, :, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], (dLdZ.shape[2]-1) // self.upsampling_factor + 1))
        dLdA = dLdZ[:, :, ::self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.A = A
        Z = np.zeros((A.shape[0], A.shape[1], (A.shape[2]-1) // self.downsampling_factor + 1))
        Z = A[:, :, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.A.shape[2]))  
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        
        Z = np.zeros((A.shape[0], A.shape[1], (A.shape[2]-1) * self.upsampling_factor + 1, (A.shape[3]-1) * self.upsampling_factor + 1))
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], (dLdZ.shape[2]-1) * self.upsampling_factor + 1, (dLdZ.shape[3]-1) * self.upsampling_factor + 1))
        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        self.A = A
        Z = np.zeros((A.shape[0], A.shape[1], A.shape[2] // self.downsampling_factor , A.shape[3] // self.downsampling_factor ))
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.A.shape[2], self.A.shape[3]))   
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ
        return dLdA
