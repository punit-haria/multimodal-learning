import tensorflow as tf
import numpy as np 

import plot
import utils

from models.joint_vae import JointVAE
from data import JointMNIST as MNIST


### PARAMETERS ###
learning_rate = 0.002                           
batch_size = 250  
z_dim = 50                
n_hidden = 200
n_paired = 1000

x_dim = 392 
y_dim = 392                   
train_steps = 10000


# data set
mnist = MNIST(n_paired)

# model
vae = JointVAE((x_dim, y_dim), z_dim, learning_rate, n_hidden, name='joint_vae')


# train model
for i in range(train_steps):
    
    # random minibatch 
    X, Y, X_joint, Y_joint = mnist.sample('train', batch_size)

    # training step
    vae.train(X, Y, X_joint, Y_joint)

    if i % 25 == 0:
        print("At iteration ", i)

        # test minibatch
        X, Y = mnist.sample('test', 1000)

        # test model
        vae.test(X, Y, X, Y)


# save final model
vae.save_state()

# reset tensorflow session and graph
vae.sess.close()
tf.reset_default_graph()
