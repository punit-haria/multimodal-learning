import numpy as np
import struct


def sample(data, batch_size):
    '''
    Generate random minibatch. 
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


def mnist():
    '''
    Load MNIST training and test sets.
    '''
    Xtr, ytr = _get_data('train')
    Xte, yte = _get_data('test')
    return Xtr, ytr, Xte, yte


def _get_data(dataset_type):
    '''
    Helper function.

    Load MNIST data.
    dataset_type: 'train' or 'test'
    '''
    if dataset_type == 'train':
        path = '../data/mnist_train'
    elif dataset_type == 'test':
        path = '../data/mnist_test'
    label_path = path + '_labels'

    data = _read_mnist(path)
    data = data / 255 
    labels = _read_mnist(label_path)

    n = len(data)
    data = data.reshape([n,-1])

    return data, labels

    
def _read_mnist(path):
    '''
    Function to read MNIST data file from 
    https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    '''
    with open(path, 'rb') as file:
        zero, dtype, dims = struct.unpack('>HBB', file.read(4))
        shape = tuple(struct.unpack('>I', file.read(4))[0] for d in range(dims))
        data = np.fromstring(file.read(), dtype=np.uint8)
        return data.reshape(shape)

    