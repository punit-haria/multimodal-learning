"""
Classes and methods to load datasets. 
"""
import numpy as np
import struct


def sample(data, batch_size):
    '''
    Generic sampling function with uniform distribution.

    data: numpy array or list of numpy arrays
    batch_size: sample size
    '''
    if not isinstance(data, list):
        n = len(data)
        idx = np.random.randint(len(data), size=batch_size)
        return idx, data[idx], 
    else:
        n = {len(x) for x in data}
        assert len(n) == 1
        n = n.pop()
        idx = np.random.randint(n, size=batch_size)
        return idx, tuple(x[idx] for x in data)




class MNIST(object):
    """
    Class to load MNIST data. 
    """
    def __init__(self,):
        self.train_path = '../data/mnist_train'
        self.test_path = '../data/mnist_test'
        self.train_labels_path = self.train_path+'_labels'
        self.test_labels_path = self.test_path+'_labels'

        self.Xtr, self.ytr = _get_data(self.train_path, self.train_labels_path)
        self.Xte, self.yte = _get_data(self.test_path, self.test_labels_path)


    def train_set(self,):
        return self.Xtr, self.ytr

    
    def test_set(self,):
        return self.Xte, self.yte


    def _get_data(data_path, labels_path):
        """
        Reads MNIST data. Rescales image pixels to be between 0 and 1. 
        """
        data = _read_mnist(data_path)
        data = data / 255
        labels = _read_mnist(labels_path)

        n = len(data)
        data = data.reshape([n, -1])

        return data, labels


    def _read_mnist(path):
        '''
        Function to read MNIST data file, taken from  
        https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
        '''
        with open(path, 'rb') as file:
            zero, dtype, dims = struct.unpack('>HBB', file.read(4))
            shape = tuple(struct.unpack('>I', file.read(4))[0] for d in range(dims))
            data = np.fromstring(file.read(), dtype=np.uint8)
            return data.reshape(shape)
        




