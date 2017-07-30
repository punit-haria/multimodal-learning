"""
Classes and methods to load datasets.
"""
import numpy as np
import struct
from scipy import ndimage
import os.path


def sample(data, batch_size):
    """
    Generic sampling function with uniform distribution.

    data: numpy array or list of numpy arrays
    batch_size: sample size
    """
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
    def __init__(self, ):
        self.train_path = '../data/mnist_train'
        self.test_path = '../data/mnist_test'
        self.train_labels_path = self.train_path + '_labels'
        self.test_labels_path = self.test_path + '_labels'

        self.Xtr, self.ytr = self._get_data(self.train_path, self.train_labels_path)
        self.Xte, self.yte = self._get_data(self.test_path, self.test_labels_path)

        self.mu = np.mean(self.Xtr, axis=0)
        self.sigma = np.std(self.Xtr, axis=0) + 1e-12


    def train_set(self, ):
        return self.Xtr, self.ytr

    def test_set(self, ):
        return self.Xte, self.yte


    def sample(self, batch_size, dtype='train', binarize=True):
        """
        Samples data from training or test set.
        """
        _, (X, Y) = self._sample(dtype, batch_size)
        if binarize:
            X = self._binarize(X)
        return X, Y


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


    def _binarize(self, data):
        """
        Samples bernoulli distribution based on pixel intensities.
        """
        return np.random.binomial(n=1, p=data)


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
        _n = len(self.Xtr)
        self.x_and_y = set(np.random.randint(_n, size=self.n_paired))
        _remain = set(np.arange(_n)) - set(self.x_and_y)
        _x_size = int(len(_remain) / 2)
        self.x_only = set(np.random.choice(list(_remain), size=_x_size, replace=False))
        self.y_only = set(np.array(list(_remain - set(self.x_only))))


    def sample(self, batch_size, dtype='train', binarize=True, include_labels=False):
        # sample naively
        idx, (batch, labels) = self._sample(dtype, batch_size)
        if binarize:
            batch = self._binarize(batch)

        # handle test set case separately
        if dtype == 'test':
            X = batch[:, 0:self.split_point]
            Y = batch[:, self.split_point:]
            if include_labels:
                return (X, labels), (Y, labels)
            else:
                return X, Y

        # separate indices into paired and missing (for training set)
        x_idx = np.array(list(set(idx) & self.x_only))
        x_idx = np.array([np.argwhere(idx == x)[0, 0] for x in x_idx], dtype=np.int32)
        y_idx = np.array(list(set(idx) & self.y_only))
        y_idx = np.array([np.argwhere(idx == x)[0, 0] for x in y_idx], dtype=np.int32)
        xy_idx = np.array(list(set(idx) & self.x_and_y))
        xy_idx = np.array([np.argwhere(idx == x)[0, 0] for x in xy_idx], dtype=np.int32)

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



class JointStratifiedMNIST(MNIST):
    """
    MNIST data treated as two output variables consisting of the top halves and bottom halves of
    each image. Sampling scheme is stratified across the paired and unpaired datasets.
    """
    def __init__(self, n_paired):
        """
        n_paired: number of paired examples (remaining examples are split into one of top or bottom halves)
        """
        super(JointStratifiedMNIST, self).__init__()  # load data
        self.n_paired = n_paired
        self.split_point = int(784 / 2)

        # joint and missing split
        _n = len(self.Xtr)
        self.x1_and_x2 = np.random.randint(_n, size=self.n_paired)
        _remain = set(np.arange(_n)) - set(self.x1_and_x2)
        _x_size = int(len(_remain) / 2)
        self.x1_only = np.random.choice(list(_remain), size=_x_size, replace=False)
        self.x2_only = np.array(list(_remain - set(self.x1_only)))

        # separate the datasets
        self.x1 = self.Xtr[self.x1_only, 0:self.split_point]
        self.y1 = self.ytr[self.x1_only]
        self.x2 = self.Xtr[self.x2_only, self.split_point:]
        self.y2 = self.ytr[self.x2_only]
        self.x12 = self.Xtr[self.x1_and_x2,:]
        self.y12 = self.ytr[self.x1_and_x2]


    def sample(self, n_paired_samples, n_unpaired_samples=250, dtype='train',
               binarize=True, include_labels=False):

        # test set case
        if dtype == 'test':
            idx, (batch, y) = sample([self.Xte, self.yte], n_paired_samples)
            if binarize:
                batch = self._binarize(batch)

            x1 = batch[:, 0:self.split_point]
            x2 = batch[:, self.split_point:]
            if include_labels:
                return (x1, y), (x2, y)
            else:
                return x1, x2

        # training set case
        elif dtype == 'train':
            n_x1 = np.random.randint(n_unpaired_samples + 1)
            n_x2 = n_unpaired_samples - n_x1

            _, (batch_p, y12) = sample([self.x12, self.y12], n_paired_samples)
            _, (x1, y1) = sample([self.x1, self.y1], n_x1)
            _, (x2, y2) = sample([self.x2, self.y2], n_x2)

            if binarize:
                batch_p = self._binarize(batch_p)
                x1 = self._binarize(x1)
                x2 = self._binarize(x2)

            x1p = batch_p[:,0:self.split_point]
            x2p = batch_p[:,self.split_point:]

            if include_labels:
                return (x1, y1), (x2, y2), (x1p, y12), (x2p, y12)
            else:
                return x1, x2, x1p, x2p



class ColouredMNIST(MNIST):
    """
    Based on dataset created in the paper: "Unsupervised Image-to-Image Translation Networks"

    X dataset consists of MNIST digits with strokes coloured as red, blue, green.
    Y dataset consists of MNIST digits transformed to an edge map, and then coloured as orange, magenta, teal.
    A small paired dataset consists of a one-to-one mapping between colours in X and colours in Y of the same
    MNIST digit.
    """
    def __init__(self, n_paired):
        """
        n_paired: number of paired examples to create
        """
        super(ColouredMNIST, self).__init__()  # load data
        self.n_paired = n_paired

        # colours for X and Y
        self.x_colours = [(255, 0, 0), (0, 219, 0), (61, 18, 198)]
        self.y_colours = [(255, 211, 0), (0, 191, 43), (0, 41, 191)]

        # load from saved if exists
        self._path = '../data/mnist_coloured.npz'
        if os.path.isfile(self._path):
            print("Loading data...", flush=True)
            data = np.load(self._path)
            self.M1 = data['arr_0']
            self.M2 = data['arr_1']
            self.M1_test = data['arr_2']
            self.M2_test = data['arr_3']
            print("Data loaded.", flush=True)

        # create modalities if data doesn't exist
        else:
            self.M1, self.M2 = self._create_modalities(self.Xtr)
            self.M1_test, self.M2_test = self._create_modalities(self.Xte)

            print("Saving data...", flush=True)
            np.savez(self._path, self.M1, self.M2, self.M1_test, self.M2_test)
            print("Saved.", flush=True)

        # separate indices
        _n = len(self.Xtr)
        self.x_and_y = set(np.random.randint(_n, size=self.n_paired))
        _remain = set(np.arange(_n)) - set(self.x_and_y)
        _x_size = int(len(_remain) / 2)
        self.x_only = set(np.random.choice(list(_remain), size=_x_size, replace=False))
        self.y_only = set(np.array(list(_remain - set(self.x_only))))

    def sample(self, batch_size=100, dtype='train', include_labels=False):
        """
        Sample minibatch.
        """
        idx, (batch, labels) = self._sample(dtype, batch_size)

        if dtype == 'test':
            X = self.M1_test[idx]
            Y = self.M2_test[idx]
            X = np.reshape(X, newshape=[-1, 784 * 3])
            Y = np.reshape(Y, newshape=[-1, 784 * 3])
            if include_labels:
                return (X, labels), (Y, labels)
            else:
                return X, Y

        else:
            # separate indices into paired and missing (for training set)
            x_idx = np.array(list(set(idx) & self.x_only))
            x_idx = np.array([np.argwhere(idx == x)[0, 0] for x in x_idx], dtype=np.int32)
            y_idx = np.array(list(set(idx) & self.y_only))
            y_idx = np.array([np.argwhere(idx == x)[0, 0] for x in y_idx], dtype=np.int32)
            xy_idx = np.array(list(set(idx) & self.x_and_y))
            xy_idx = np.array([np.argwhere(idx == x)[0, 0] for x in xy_idx], dtype=np.int32)

            # create separate arrays for jointly observed and marginal data
            X = self.M1[x_idx]
            Y = self.M2[y_idx]
            X_joint = self.M1[xy_idx]
            Y_joint = self.M2[xy_idx]

            # reshape
            X = np.reshape(X, newshape=[-1, 784 * 3])
            Y = np.reshape(Y, newshape=[-1, 784 * 3])
            X_joint = np.reshape(X_joint, newshape=[-1, 784 * 3])
            Y_joint = np.reshape(Y_joint, newshape=[-1, 784 * 3])

            if include_labels:  # split label data too
                lX = labels[x_idx]
                lY = labels[y_idx]
                l_joint = labels[xy_idx]
                return (X_joint, l_joint), (Y_joint, l_joint), (X, lX), (Y, lY)
            else:
                return X_joint, Y_joint, X, Y


    def _create_modalities(self, data):
        """
        Creates X and Y datasets from input MNIST data.

        data: numpy array of MNIST digits, with dimensions: #digits x 784
        """
        # randomly assign colours
        x_bank, y_bank = self._sample_random_colours(len(data))

        # colour digits
        print("Colouring modalities...", flush=True)
        X = self._colour(data, x_bank)
        Y = self._colour(data, y_bank)

        # reshape and scale
        X = np.reshape(X, newshape=[-1, 28, 28, 3]) / 255
        Y = np.reshape(Y, newshape=[-1, 28, 28, 3])  # normalized in _edge_map

        # compute edge map
        print("Computing edge map...", flush=True)
        Y = self._edge_map(Y)

        return X, Y

    def _edge_map(self, data):
        """
        Converts MNIST digits into corresponding edge map.

        data: numpy array of MNIST digits, with dimensions: #images x height x width
        """
        n = len(data)
        edges = np.zeros(shape=data.shape)
        for i in range(n):
            im = data[i]
            sx = ndimage.sobel(im, axis=0, mode='constant')
            sy = ndimage.sobel(im, axis=1, mode='constant')
            sob = np.hypot(sx, sy)
            _max = np.max(sob)
            edges[i] = sob / _max

        return edges

    def _colour(self, data, colours):
        """
        Randomly colours MNIST digits into one of 3 colours.

        data: numpy array of MNIST digits, with dimensions: #images x 784
        colours: numpy array of colours, with dimensions: #images x 3
        """
        rgb = []
        for i in range(3):
            rgb_comp = np.zeros(data.shape)
            for j in range(len(data)):
                ones = np.where(data[j] > 0)[0]
                rgb_comp[j] = data[j]
                rgb_comp[j, ones] = colours[j, i]
            rgb.append(rgb_comp)

        return np.stack(rgb, axis=-1)

    def _sample_random_colours(self, n_samples):
        """
        Draws random colours from each colour bank.

        n_samples: number of random colours to draw
        """
        x_bank = np.array(self.x_colours)
        y_bank = np.array(self.y_colours)
        idx = np.random.randint(len(x_bank), size=n_samples)

        return x_bank[idx], y_bank[idx]



class ColouredStratifiedMNIST(ColouredMNIST):
    """
    Based on dataset created in the paper: "Unsupervised Image-to-Image Translation Networks"

    X dataset consists of MNIST digits with strokes coloured as red, blue, green.
    Y dataset consists of MNIST digits transformed to an edge map, and then coloured as orange, magenta, teal.
    A small paired dataset consists of a one-to-one mapping between colours in X and colours in Y of the same
    MNIST digit.
    """
    def __init__(self, n_paired):
        """
        n_paired: number of paired examples to create
        """
        super(ColouredStratifiedMNIST, self).__init__(n_paired)  # load data

        self.x1_and_x2 = np.array(list(self.x_and_y))
        self.x1_only = np.array(list(self.x_only))
        self.x2_only = np.array(list(self.y_only))

        # separate the datasets
        self.x1 = self.M1[self.x1_only]
        self.y1 = self.ytr[self.x1_only]
        self.x2 = self.M2[self.x2_only]
        self.y2 = self.ytr[self.x2_only]
        self.x1p = self.M1[self.x1_and_x2]
        self.x2p = self.M2[self.x1_and_x2]
        self.yp = self.ytr[self.x1_and_x2]


    def sample_stratified(self, n_paired_samples, n_unpaired_samples=250, dtype='train', include_labels=False):
        # test set case
        if dtype == 'test':

            _, (x1, x2, y) = sample([self.M1_test, self.M2_test, self.yte], n_paired_samples)

            # reshape
            x1 = np.reshape(x1, newshape=[-1, 784 * 3])
            x2 = np.reshape(x2, newshape=[-1, 784 * 3])

            if include_labels:
                return (x1, y), (x2, y)
            else:
                return x1, x2

        # training set case
        elif dtype == 'train':
            n_x1 = np.random.randint(n_unpaired_samples + 1)
            n_x2 = n_unpaired_samples - n_x1

            _, (x1p, x2p, yp) = sample([self.x1p, self.x2p, self.yp], n_paired_samples)
            _, (x1, y1) = sample([self.x1, self.y1], n_x1)
            _, (x2, y2) = sample([self.x2, self.y2], n_x2)

            # reshape
            x1 = np.reshape(x1, newshape=[-1, 784 * 3])
            x2 = np.reshape(x2, newshape=[-1, 784 * 3])
            x1p = np.reshape(x1p, newshape=[-1, 784 * 3])
            x2p = np.reshape(x2p, newshape=[-1, 784 * 3])

            if include_labels:
                return (x1, y1), (x2, y2), (x1p, yp), (x2p, yp)
            else:
                return x1, x2, x1p, x2p




class CIFAR(object):

    def __init__(self, ):
        self.xtr, self.ytr, self.xte, self.yte = self._get_data()
        self.xtr = self.xtr / 255
        self.xte = self.xte / 255


    def train_set(self, ):
        return self.xtr, self.ytr

    def test_set(self, ):
        return self.xte, self.yte


    def sample(self, batch_size, dtype='train'):
        """
        Samples data from training or test set.
        """
        _, (X, Y) = self._sample(dtype, batch_size)

        return X, Y


    def _sample(self, dtype='train', batch_size=100):
        """
        Samples data from training set.
        """
        if dtype == 'train':
            return sample([self.xtr, self.ytr], batch_size)
        elif dtype == 'test':
            return sample([self.xte, self.yte], batch_size)
        else:
            raise Exception('Training or test set not selected..')


    def _get_data(self, ):

        prefix = "../data/cifar-10/"

        xtr = []
        ytr = []
        for i in range(1,6):
            path = prefix + "data_batch_" + str(i)
            x, y = self._unpickle(path)

            xtr.append(x)
            ytr.extend(y)

        xtr = np.concatenate(xtr, axis=0)
        xtr = self._transpose(xtr)
        ytr = np.array(ytr)

        path = prefix + "test_batch"
        xte, yte = self._unpickle(path)
        xte = self._transpose(xte)
        yte = np.array(yte)

        return xtr, ytr, xte, yte


    def _transpose(self, x):

        x = np.reshape(x, newshape=[-1, 3, 32, 32])
        x = np.transpose(x, axes=(0,2,3,1))
        x = np.reshape(x, newshape=[-1, 3072])

        return x


    def _unpickle(self, f_name):
        import pickle
        with open(f_name, 'rb') as fo:
            dd = pickle.load(fo, encoding='bytes')
        return dd[b'data'], dd[b'labels']