import numpy as np 

import plot
import utils

from data import ColouredMNIST as MNIST



n_paired = 1000
batch_size = 10000

mnist = MNIST(n_paired)

X, Y, X_joint, Y_joint = mnist.sample('train', batch_size)

#X, Y = mnist.sample('test', 1000)

X = X_joint[0:20]
Y = Y_joint[0:20]
images = np.concatenate([X,Y], axis=0)

plot.plot_images(images, 8, 5, '../plots/testing')

