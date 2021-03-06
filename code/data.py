"""
Classes and methods to load datasets.
"""
import numpy as np
import struct
from scipy.misc import imresize
from scipy import ndimage
import os
import os.path
import pandas as pd
import json
from collections import defaultdict
from pathlib import Path as pathlib_path
import pickle

'''

Contains helper methods and classes for loading each dataset.
 
'''


def sample(data, batch_size):
    """
    Generic sampling function with uniform distribution.

    data: numpy array or list of numpy arrays
    batch_size: sample size
    """
    if not isinstance(data, list):
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


    def sample_stratified(self, n_paired_samples, n_unpaired_samples=250, dtype='train',
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
            n_min = 2 * n_unpaired_samples // 5
            n_min = max(1, n_min)
            n_max = n_unpaired_samples - n_min

            n_x1 = np.random.randint(low=n_min, high=n_max + 1)
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
                return (X, lX), (Y, lY), (X_joint, l_joint), (Y_joint, l_joint)
            else:
                return X, Y, X_joint, Y_joint


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
    def __init__(self, n_paired, censor=False):
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


        if censor:

            numbers_train = [0,1,2,3,4,5,6,7]
            numbers_test = [8,9]

            idx = []
            for i, ix in enumerate(self.y1):
                if ix in numbers_train:
                    idx.append(i)
            self.y1 = self.y1[idx]
            self.x1 = self.x1[idx]

            idx = []
            for i, ix in enumerate(self.y2):
                if ix in numbers_train:
                    idx.append(i)
            self.y2 = self.y2[idx]
            self.x2 = self.x2[idx]

            idx = []
            for i, ix in enumerate(self.yp):
                if ix in numbers_train:
                    idx.append(i)
            self.yp = self.yp[idx]
            self.x1p = self.x1p[idx]
            self.x2p = self.x2p[idx]

            idx = []
            for i, ix in enumerate(self.yte):
                if ix in numbers_test:
                    idx.append(i)
            self.yte = self.yte[idx]
            self.M1_test = self.M1_test[idx]
            self.M2_test = self.M2_test[idx]



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
            n_min = 2 * n_unpaired_samples // 5
            n_min = max(1, n_min)
            n_max = n_unpaired_samples - n_min

            n_x1 = np.random.randint(low=n_min, high=n_max + 1)
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




class Sketches(object):

    def __init__(self, n_paired):

        _raw_photo_path = '../data/sketchy/256x256/photo/tx_000100000000/'
        _raw_sketch_path = '../data/sketchy/256x256/sketch/tx_000100000000/'

        _data_path = '../data/sketch.npz'

        if os.path.isfile(_data_path):  # load processed data

            print("Loading data...", flush=True)

            data = np.load(_data_path)
            self.x1 = data['arr_0']
            self.x2 = data['arr_1']
            self.ytr = data['arr_2']
            self.x1_test = data['arr_3']
            self.x2_test = data['arr_4']
            self.yte = data['arr_5']

            print("Data loaded.", flush=True)

        else: # process data and load
            x1 = []
            x2 = []
            y = []
            train = []
            test = []

            print("Processing data..", flush=True)

            categories = [p for p in os.listdir(_raw_photo_path)
                         if os.path.isdir(os.path.join(_raw_photo_path, p))]

            i = 0

            for cat in categories:
                print("At category: ", cat, flush=True)

                cat_photo_path = _raw_photo_path + cat + '/'
                cat_sketch_path = _raw_sketch_path + cat + '/'

                photo_files = [p for p in os.listdir(cat_photo_path)
                               if os.path.isfile(os.path.join(cat_photo_path, p))]

                sketch_files = [p for p in os.listdir(cat_sketch_path)
                                if os.path.isfile(os.path.join(cat_sketch_path, p))]

                for f in photo_files:
                    photo_path = cat_photo_path + f

                    photo = ndimage.imread(photo_path)
                    photo = imresize(photo, size=0.25, interp='cubic')
                    photo = np.reshape(photo, newshape=[1, -1])

                    sketches = [p for p in sketch_files if f.replace('.jpg','')+'-' in p]

                    is_train = np.random.binomial(n=1, p=0.85)  # sort into train/test sets

                    for sk in sketches:
                        sketch_path = cat_sketch_path + sk

                        sketch = ndimage.imread(sketch_path)
                        sketch = imresize(sketch, size=0.25, interp='cubic')
                        sketch = np.reshape(sketch, newshape=[1, -1])

                        x1.append(photo)
                        x2.append(sketch)
                        y.append(cat)

                        if is_train == 1:
                            train.append(i)
                        else:
                            test.append(i)

                        i += 1

            y = pd.Series(y)
            y = pd.Categorical(y)
            y = y.codes

            assert len(x1) == len(x2)
            x1 = np.concatenate(x1, axis=0)
            x2 = np.concatenate(x2, axis=0)

            print("x1 shape: ", x1.shape, flush=True)
            print("x2 shape: ", x2.shape, flush=True)

            self.x1 = x1[train]
            self.x2 = x2[train]
            self.ytr = y[train]
            self.x1_test = x1[test]
            self.x2_test = x2[test]
            self.yte = y[test]

            print("Saving data...", flush=True)
            np.savez(_data_path, self.x1, self.x2, self.ytr, self.x1_test, self.x2_test, self.yte)
            print("Saved.", flush=True)


        # construct pairings
        _n = len(self.x1)
        self.x1_and_x2 = set(np.random.randint(_n, size=n_paired))
        _remain = set(np.arange(_n)) - set(self.x1_and_x2)
        _x_size = int(len(_remain) / 2)
        self.x1_only = set(np.random.choice(list(_remain), size=_x_size, replace=False))
        self.x2_only = set(np.array(list(_remain - set(self.x1_only))))

        self.x1_and_x2 = np.array(list(self.x1_and_x2))
        self.x1_only = np.array(list(self.x1_only))
        self.x2_only = np.array(list(self.x2_only))

        # separate out datasets
        self.x1u = self.x1[self.x1_only]
        self.y1u = self.ytr[self.x1_only]
        self.x2u = self.x2[self.x2_only]
        self.y2u = self.ytr[self.x2_only]
        self.x1p = self.x1[self.x1_and_x2]
        self.x2p = self.x2[self.x1_and_x2]
        self.yp = self.ytr[self.x1_and_x2]


    def sample_stratified(self, n_paired_samples, n_unpaired_samples=250, dtype='train', include_labels=False):
        # test set case
        if dtype == 'test':

            _, (x1, x2, y) = sample([self.x1_test, self.x2_test, self.yte], n_paired_samples)

            x1 = x1 / 255
            x2 = x2 / 255

            if include_labels:
                return (x1, y), (x2, y)
            else:
                return x1, x2

        # training set case
        elif dtype == 'train':
            n_min = 2 * n_unpaired_samples // 5
            n_min = max(1, n_min)
            n_max = n_unpaired_samples - n_min

            n_x1 = np.random.randint(low=n_min, high=n_max + 1)
            n_x2 = n_unpaired_samples - n_x1

            _, (x1p, x2p, yp) = sample([self.x1p, self.x2p, self.yp], n_paired_samples)
            _, (x1, y1) = sample([self.x1u, self.y1u], n_x1)
            _, (x2, y2) = sample([self.x2u, self.y2u], n_x2)

            x1 = x1 / 255
            x2 = x2 / 255
            x1p = x1p / 255
            x2p = x2p / 255

            if include_labels:
                return (x1, y1), (x2, y2), (x1p, yp), (x2p, yp)
            else:
                return x1, x2, x1p, x2p



class DayNight(object):

    def __init__(self,):

        data_path = '../data/dnim.npz'

        if os.path.isfile(data_path):  # load processed data

            print("Loading data...", flush=True)

            data = np.load(data_path)
            self.x1p = data['arr_0']
            self.x2p = data['arr_1']
            self.yp = data['arr_2']
            self.x1 = data['arr_3']
            self.x2 = data['arr_4']
            self.y1 = data['arr_5']
            self.y2 = data['arr_6']
            self.x1_test = data['arr_7']
            self.x2_test = data['arr_8']
            self.y_test = data['arr_9']

            print("Data loaded.", flush=True)

        else: # process data and load

            dnim_path = '../data/dnim/Image/'
            dnim_stamps_path = '../data/dnim/time_stamp/'

            print("Processing data..", flush=True)

            dnim_stamps = [p for p in os.listdir(dnim_stamps_path)
                           if os.path.isfile(os.path.join(dnim_stamps_path, p))]

            df = []

            for i, st in enumerate(dnim_stamps):
                path = dnim_stamps_path + st

                tst = pd.read_csv(path, sep=' ', header=None, names=['f_name', 'date', 'h', 'm'])

                tst['camera'] = [st.replace('.txt','')] * len(tst)

                # train/test indicator
                is_train = [1] * len(tst) if i < 11 else [0] * len(tst)
                tst['is_train'] = pd.Series(is_train)

                df.append(tst)

            df = pd.concat(df, ignore_index=True)

            night = [23,0,1,2,3]
            day = [9,10,11,12,13,14,15]

            pairs = []
            names = ['camera', 'is_train', 'day_file', 'night_file']

            print("Constructing pairings...", flush=True)

            for _, rowd in df.iterrows():
                cam = rowd['camera']
                d = rowd['h']

                if d in day:
                    for _, rown in df[df['camera'] == cam].iterrows():

                        assert cam == rown['camera']

                        n = rown['h']
                        if n in night:
                            pairs.append([cam, rowd['is_train'], rowd['f_name'], rown['f_name']])

            pairs = pd.DataFrame(pairs, columns=names)

            x1 = []
            x2 = []
            y = []
            train = []
            test = []

            print("Processing DNIM images...", flush=True)

            i = 0
            for _, row in pairs.iterrows():

                if i % 1000 == 0:
                    print("At row: ", i, flush=True)

                cam = row['camera']

                day_path = dnim_path + cam + '/' + row['day_file']
                night_path = dnim_path + cam + '/' + row['night_file']

                day = ndimage.imread(day_path)
                day = imresize(day, size=(44,64), interp='cubic')
                day = np.reshape(day, newshape=[1, -1])

                night = ndimage.imread(night_path)
                night = imresize(night, size=(44,64), interp='cubic')
                night = np.reshape(night, newshape=[1, -1])

                x1.append(day)
                x2.append(night)
                y.append(cam)

                if row['is_train'] == 1:
                    train.append(i)
                else:
                    test.append(i)
                i += 1

            y = pd.Series(y)
            y = pd.Categorical(y)
            y = y.codes

            assert len(x1) == len(x2)
            x1 = np.concatenate(x1, axis=0)
            x2 = np.concatenate(x2, axis=0)

            self.x1p = x1[train]
            self.x2p = x2[train]
            self.yp = y[train]
            self.x1_test = x1[test]
            self.x2_test = x2[test]
            self.y_test = y[test]

            # add unsupervised data (amos)

            amos_path = '../data/amos/'

            amos_cams = [p for p in os.listdir(amos_path)
                         if os.path.isdir(os.path.join(amos_path, p))]

            x1 = []
            x2 = []
            y1 = []
            y2 = []

            night = [23, 0, 1, 2, 3]
            day = [9, 10, 11, 12, 13, 14, 15]

            print("Processing AMOS data...", flush=True)

            n_fails = 0

            for cam in amos_cams:
                cam_path = amos_path + cam + '/2016.08/'

                print("At camera: ", cam, flush=True)

                ims = [p for p in os.listdir(cam_path)
                           if os.path.isfile(os.path.join(cam_path, p))]

                print(len(ims))

                for f in ims:
                    loc = f.index('_')
                    hour = int(f[loc+1:loc+3])

                    f_path = cam_path + f

                    try:
                        if hour in day:

                            image = ndimage.imread(f_path)
                            image = imresize(image, size=(44, 64), interp='cubic')
                            image = np.reshape(image, newshape=[1, -1])

                            x1.append(image)
                            y1.append(cam)

                        elif hour in night:

                            image = ndimage.imread(f_path)
                            image = imresize(image, size=(44, 64), interp='cubic')
                            image = np.reshape(image, newshape=[1, -1])

                            x2.append(image)
                            y2.append(cam)

                    except:
                        print("Error at: ", f_path, flush=True)
                        n_fails += 1


            print("Number of Failures: ", n_fails, flush=True)

            y1 = pd.Series(y1)
            y1 = pd.Categorical(y1)
            self.y1 = y1.codes

            y2 = pd.Series(y2)
            y2 = pd.Categorical(y2)
            self.y2 = y2.codes

            self.x1 = np.concatenate(x1, axis=0)
            self.x2 = np.concatenate(x2, axis=0)


            print("Unpaired x1: ", self.x1.shape, flush=True)
            print("Unpaired x2: ", self.x2.shape, flush=True)

            print("Paired x1: ", self.x1p.shape, flush=True)
            print("Paired x2: ", self.x2p.shape, flush=True)

            print("Saving data...", flush=True)
            np.savez(data_path, self.x1p, self.x2p, self.yp, self.x1, self.x2, self.y1, self.y2,
                     self.x1_test, self.x2_test, self.y_test)
            print("Saved.", flush=True)


    def sample_stratified(self, n_paired_samples, n_unpaired_samples=250, dtype='train', include_labels=False):
        # test set case
        if dtype == 'test':

            _, (x1, x2, y) = sample([self.x1_test, self.x2_test, self.y_test], n_paired_samples)

            x1 = x1 / 255
            x2 = x2 / 255

            if include_labels:
                return (x1, y), (x2, y)
            else:
                return x1, x2

        # training set case
        elif dtype == 'train':
            n_min = 2 * n_unpaired_samples // 5
            n_min = max(1, n_min)
            n_max = n_unpaired_samples - n_min

            n_x1 = np.random.randint(low=n_min, high=n_max + 1)
            n_x2 = n_unpaired_samples - n_x1

            _, (x1p, x2p, yp) = sample([self.x1p, self.x2p, self.yp], n_paired_samples)
            _, (x1, y1) = sample([self.x1, self.y1], n_x1)
            _, (x2, y2) = sample([self.x2, self.y2], n_x2)

            x1 = x1 / 255
            x2 = x2 / 255
            x1p = x1p / 255
            x2p = x2p / 255

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



class MSCOCO(object):

    def __init__(self, n_paired):

        _train_annotations_path = '../data/mscoco/annotations/captions_train2014.json'
        _val_annotations_path = '../data/mscoco/annotations/captions_val2014.json'

        _train_images_dir = '../data/mscoco/train2014/'
        _val_images_dir = '../data/mscoco/val2014/'

        _caption_path = '../data/mscoco/captions.pickle'
        _image_path = '../data/mscoco/images.pickle'

        self._padding = '<PAD>'
        self._oov = '<OOV>'
        self._go = '<GO>'
        self._eof = '<EOF>'
        self._symbols = [self._oov, self._padding, self._eof, self._go]
        self._inverse_vocab = None

        paths = [(_train_annotations_path, _train_images_dir), (_val_annotations_path, _val_images_dir)]

        if os.path.isfile(_image_path):

            print("Loading images...", flush=True)

            with open(_image_path, 'rb') as ff:
                data = pickle.load(ff)

            self._images = data['images']
            self._val_images = data['val_images']

            print("Images loaded.", flush=True)

        else:

            for j, (ann_p,im_p) in enumerate(paths):

                with open(ann_p) as ff:
                    ann = json.load(ff)

                print("Creating image dictionary..", flush=True)
                images = dict()  # key,value ---> image_id, image array
                for k in ann['images']:
                    file_path = im_p + k['file_name']
                    im_file = pathlib_path(file_path)
                    if im_file.exists():
                        image = ndimage.imread(file_path)
                        image = imresize(image, size=(48, 64), interp='cubic')

                        if image.shape == (48, 64):
                            image = np.expand_dims(image, axis=2)
                            image = np.concatenate((image, image, image), axis=2)

                        image = np.reshape(image, newshape=[1, -1])

                        images[k['id']] = image

                if j == 0:      # training set
                    self._images = images
                else:           # validation set
                    self._val_images = images

            tosave = dict()
            tosave['images'] = self._images
            tosave['val_images'] = self._val_images

            print("Saving images...", flush=True)
            with open(_image_path, 'wb') as ff:
                pickle.dump(tosave, ff, pickle.HIGHEST_PROTOCOL)
            print("Saved.", flush=True)


        if os.path.isfile(_caption_path):  # load processed data

            print("Loading data...", flush=True)

            with open(_caption_path, 'rb') as ff:
                data = pickle.load(ff)

            self._vocab = data['vocab']

            self._captions = data['captions']
            self._imcapt = data['imcapt']
            self._val_captions = data['val_captions']
            self._val_imcapt = data['val_imcapt']
            self._max_seq_len = data['max_seq_len']

            print("Data loaded.", flush=True)


        else: # process data and load
            print("Processing data..", flush=True)

            self._max_seq_len = 1

            for j, (ann_p,im_p) in enumerate(paths):

                with open(ann_p) as ff:
                    ann = json.load(ff)

                print("Creating caption dictionary..", flush=True)
                captions = dict()   # key,value ---> caption_id, word sequence
                for k in ann['annotations']:
                    capt = k['caption']

                    # caption preprocessing
                    capt = capt.strip()     # remove unnecessary whitespace
                    capt = capt.lower()     # make lower case
                    capt = capt.replace('.', ' ')  # remove periods
                    capt = capt.replace(',', ' ')  # remove commas
                    capt = capt.replace('?', ' ')  # remove question marks
                    capt = capt.replace('-', ' ')  # remove dashes
                    capt = capt.replace('"', ' " ')  # expand double quotes
                    capt = capt.replace('(', ' ( ')  # expand brackets
                    capt = capt.replace(')', ' ) ')  # expand brackets
                    capt = capt.replace('{', ' { ')  # expand brackets
                    capt = capt.replace('}', ' } ')  # expand brackets
                    capt = capt.split()  # split string
                    capt.append(self._eof)  # pad with EOF character

                    captions[k['id']] = capt


                self._max_seq_len = max(max([len(_v) for _,_v in captions.items()]), self._max_seq_len)
                print("Max sequence length: ", self._max_seq_len, flush=True)


                if j == 0: # training set
                    print("Word frequencies", flush=True)
                    freqs = defaultdict(int)
                    for _, capt in captions.items():
                        for word in capt:
                            freqs[word] += 1

                    print("Adding <OOV> words", flush=True)
                    min_freq = 2                # minimum word frequency
                    for k,capt in captions.items():
                        for i,w in enumerate(capt):
                            if freqs[w] < min_freq:
                                if np.random.binomial(n=1, p=0.9) == 1:   # 90% chance of setting <OOV>
                                    capt[i] = self._oov


                print("Creating vocabulary..", flush=True)
                if j > 0: # validation set
                    vocab = self._vocab

                else:
                    vocab = dict()  # key,value ---> word, word_id
                    words = {w for _, _v in captions.items() for w in _v}
                    for i,w in enumerate(words):
                        vocab[w] = i

                    for s in self._symbols:  # add symbols to vocab dictionary if not already there
                        if s not in vocab:
                            idx = max([v for k,v in vocab.items()]) + 1
                            vocab[s] = idx


                print("Converting captions to ids (from vocab)..", flush=True)
                for _k,_v in captions.items():
                    for i in range(len(_v)):
                        if _v[i] in vocab:
                            _v[i] = vocab[_v[i]]
                        else:
                            _v[i] = vocab[self._oov]


                print("Creating image-caption mapping..", flush=True)
                im_capt = defaultdict(set)    # key,value ---> image_id, set of caption ids
                for k in ann['annotations']:
                    im_capt[k['image_id']].add(k['id'])


                if j == 0:      # training set
                    self._captions = captions
                    self._vocab = vocab
                    self._imcapt = im_capt

                else:           # validation set
                    self._val_captions = captions
                    self._vocab = vocab
                    self._val_imcapt = im_capt

            tosave = dict()
            tosave['vocab'] = self._vocab

            tosave['captions'] = self._captions
            tosave['imcapt'] = self._imcapt
            tosave['val_captions'] = self._val_captions
            tosave['val_imcapt'] = self._val_imcapt
            tosave['max_seq_len'] = self._max_seq_len

            print("Saving data...", flush=True)
            with open(_caption_path, 'wb') as ff:
                pickle.dump(tosave, ff, pickle.HIGHEST_PROTOCOL)
            print("Saved.", flush=True)

        # lists of image ids
        self.image_ids = list(self._images.keys())
        self.val_image_ids = list(self._val_images.keys())

        # construct pairings
        _n = len(self.image_ids)
        self.paired = set(np.random.choice(self.image_ids, size=n_paired, replace=False))
        _remain = set(self.image_ids) - self.paired
        _each_size = len(_remain) // 2
        self.image_only = set(np.random.choice(list(_remain), size=_each_size, replace=False))
        self.caption_only = _remain - self.image_only

        self.paired = list(self.paired)
        self.image_only = list(self.image_only)
        self.caption_only = list(self.caption_only)


    def get_max_seq_len(self):
        return self._max_seq_len

    def get_vocab_size(self):
        return len(self._vocab)

    def get_word(self, word_id):
        if self._inverse_vocab is None:
            self._inverse_vocab = {v: k for k, v in self._vocab.items()}
        return self._inverse_vocab[word_id]


    def _sample_setup(self, image_ids, train):
        """
        Generate samples in matrix form based on already sampled images.
        """
        if train:
            imcapt = self._imcapt
            captions = self._captions
            images = self._images

        else:
            imcapt = self._val_imcapt
            captions = self._val_captions
            images = self._val_images

        x_caption = []
        x_caption_decode = []
        x_image = []
        seq_lens = []
        for i in image_ids:
            capts = imcapt[i]
            capt_id = int(np.random.choice(list(capts), size=1))
            caption = captions[capt_id]

            seq_lens.append(len(caption))  # true sequence length

            # add padding to each caption
            while len(caption) < self._max_seq_len:
                caption.append(self._vocab[self._padding])

            x_caption.append(caption)

            caption_dec = [self._vocab[self._go]] + list(caption)
            x_caption_decode.append(caption_dec)

            image = images[i]

            x_image.append(image)

        x_image = np.array(x_image) / 255
        x_image = np.squeeze(x_image)

        x_caption = np.array(x_caption)

        x_caption_decode = np.array(x_caption_decode)
        x_caption_decode = x_caption_decode[:,:-1]

        seq_lens = np.array(seq_lens)

        return x_image, x_caption, seq_lens, x_caption_decode


    def sample_stratified(self, n_paired_samples, n_unpaired_samples=128, dtype='train'):

        # note: decoder input begins with <GO> symbol

        # test set case
        if dtype == 'test':

            ids = list(np.random.choice(self.val_image_ids, size=n_paired_samples, replace=False))
            xi, xc, sl, xc_dec = self._sample_setup(ids, train=False)

            return xi, xc, sl, xc_dec

        # training set case
        elif dtype == 'train':

            n_min = 2 * n_unpaired_samples // 5
            n_min = max(1, n_min)
            n_max = n_unpaired_samples - n_min

            n_x1 = np.random.randint(low=n_min, high=n_max + 1)
            n_x2 = n_unpaired_samples - n_x1

            paired_ids = list(np.random.choice(self.paired, size=n_paired_samples, replace=False))
            xpi, xpc, slp, xpc_dec = self._sample_setup(paired_ids, train=True)

            image_only_ids = list(np.random.choice(self.image_only, size=n_x1, replace=False))
            xi, _, _, _ = self._sample_setup(image_only_ids, train=True)

            caption_only_ids = list(np.random.choice(self.caption_only, size=n_x2, replace=False))
            _, xc, sl, xc_dec = self._sample_setup(caption_only_ids, train=True)

            return xi, xc, sl, xc_dec, xpi, xpc, slp, xpc_dec



