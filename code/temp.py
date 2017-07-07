import numpy as np
from data import ColouredMNIST as MNIST
import plot

mnist = MNIST(1000)

X, Y = mnist.sample('test', 18)

images = np.concatenate((X,Y),axis=0)
images = np.reshape(images, [-1,28,28,3])

plot.plot_images(images, 6, 6, '/Users/punit/Dropbox/ucl/thesis/presentation/generated_mnist_colours.png')


