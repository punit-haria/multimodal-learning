"""
Classes and methods to load datasets. 
"""
import numpy as np
import struct



class DAY_NIGHT(object):
    """
    Class to load day/night data.
    """
    def __init__(self,):
        pass



class MNIST(object):
    """
    Class to load MNIST data. 
    """
    def __init__(self,):
        self.train_path = '../data/mnist_train'
        self.test_path = '../data/mnist_test'
        self.train_labels_path = self.train_path+'_labels'
        self.test_labels_path = self.test_path+'_labels'

        self.Xtr, self.ytr = self._get_data(self.train_path, self.train_labels_path)
        self.Xte, self.yte = self._get_data(self.test_path, self.test_labels_path)


    def sample(self, dtype='train', batch_size=100):
        """
        Samples data from training set. 
        """
        _, (X,Y) = self._sample(dtype, batch_size)
        return X, Y


    def train_set(self,):
        return self.Xtr, self.ytr

    
    def test_set(self,):
        return self.Xte, self.yte


    def _sample(self, dtype='train', batch_size=100):
        """
        Samples data from training set. 
        """
        if dtype == 'train':
            return sample([self.Xtr, self.ytr], batch_size)
        elif dtype == 'test':
            return sample([self.Xte, self.yte], batch_size)
        else:
            raise Exception('Training or test set not selected..')


    def _get_data(self, data_path, labels_path):
        """
        Reads MNIST data. Rescales image pixels to be between 0 and 1. 
        """
        data = self._read_mnist(data_path)
        data = data / 255
        labels = self._read_mnist(labels_path)

        n = len(data)
        data = data.reshape([n, -1])

        return data, labels


    def _read_mnist(self, path):
        '''
        Function to read MNIST data file, taken from  
        https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
        '''
        with open(path, 'rb') as file:
            zero, dtype, dims = struct.unpack('>HBB', file.read(4))
            shape = tuple(struct.unpack('>I', file.read(4))[0] for d in range(dims))
            data = np.fromstring(file.read(), dtype=np.uint8)
            return data.reshape(shape)
        


class JointMNIST(MNIST):
    """
    MNIST data treated as two output variables consisting of the top halves and bottom halves of 
    each image. 
    """
    def __init__(self, n_paired):
        """
        n_paired: number of paired examples (remaining examples are split into one of top or bottom halves)
        """
        super(JointMNIST, self).__init__()  # load data
        self.n_paired = n_paired  
        self.split_point = int(784 / 2)

        # joint and missing split
        self.x_and_y = set(np.arange(1000))
        self.x_only = set(len(self.x_and_y) + np.arange(29500))
        self.y_only = set(len(self.x_and_y) + len(self.x_only) + np.arange(29500))


    def sample(self, dtype='train', batch_size=100, include_labels=False):
        # sample naively
        idx, (batch,labels) = self._sample(dtype, batch_size)

        # handle test set case separately
        if dtype == 'test':
            X = batch[:, 0:self.split_point]
            Y = batch[:,self.split_point:]
            if include_labels:
                return (X,labels), (Y,labels)
            else:
                return X, Y

        # separate indices into paired and missing (for training set)
        x_idx = np.array(list(set(idx) & self.x_only))
        x_idx = np.array([np.argwhere(idx == x)[0,0]  for x in x_idx], dtype=np.int32)
        y_idx = np.array(list(set(idx) & self.y_only))
        y_idx = np.array([np.argwhere(idx == x)[0,0]  for x in y_idx], dtype=np.int32)
        xy_idx = np.array(list(set(idx) & self.x_and_y))
        xy_idx = np.array([np.argwhere(idx == x)[0,0]  for x in xy_idx], dtype=np.int32)

        # create separate arrays for jointly observed and marginal data
        X = batch[x_idx, 0:self.split_point]
        Y = batch[y_idx, self.split_point:]
        X_joint = batch[xy_idx, 0:self.split_point]
        Y_joint = batch[xy_idx, self.split_point:]
    
        if include_labels:  # split label data too
            lX = labels[x_idx]
            lY = labels[y_idx]
            l_joint = labels[xy_idx]
            return (X, lX), (Y, lY), (X_joint, l_joint), (Y_joint, l_joint)

        else:
            return X, Y, X_joint, Y_joint




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
