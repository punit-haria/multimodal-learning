import tensorflow as tf
import numpy as np 

import components as cp
import data
import plot
import utils

from models.joint_vae import JointVAE


# parameters
learning_rate = 0.002                           
batch_size = 250  
x_dim = 392 
y_dim = 392                   
z_dim = 50                    
train_steps = 10000


# joint/missing split
x_and_y = set(np.arange(1000))
x_only = set(len(x_and_y) + np.arange(29500))
y_only = set(len(x_and_y) + len(x_only) + np.arange(29500))


# train/test sets
Xtr, ytr, Xte, yte = data.mnist()


# joint variational auto-encoder
model_name = 'vae_lr_'+str(learning_rate)+'_batch_'+str(batch_size)
vae = JointVAE((x_dim, y_dim), z_dim, learning_rate, name=model_name)

# train model
for i in range(train_steps):
    
    # randomly sampled minibatch 
    idx, batch = data.sample(Xtr, batch_size)

    # separate indices
    x_idx = np.array(list(set(idx) & x_only))
    x_idx = np.array([np.argwhere(idx == x)[0,0]  for x in x_idx], dtype=np.int32)
    y_idx = np.array(list(set(idx) & y_only))
    y_idx = np.array([np.argwhere(idx == x)[0,0]  for x in y_idx], dtype=np.int32)
    xy_idx = np.array(list(set(idx) & x_and_y))
    xy_idx = np.array([np.argwhere(idx == x)[0,0]  for x in xy_idx], dtype=np.int32)

    # separate jointly observed and missing data
    X = batch[x_idx, 0:x_dim]
    Y = batch[y_idx, x_dim:]
    X_joint = batch[xy_idx, 0:x_dim]
    Y_joint = batch[xy_idx, x_dim:]

    # training step
    vae.train(X, Y, X_joint, Y_joint)


    if i % 25 == 0:
        print("At iteration ", i)

        # test minibatch
        idx, test_batch = data.sample(Xte, 1000)

        # test solely on joint data
        X = test_batch[:,0:x_dim]
        Y = test_batch[:,x_dim:]

        # test model
        vae.test(X, Y, X, Y)


# save final model
vae.save_state()

# reset tensorflow session and graph
vae.sess.close()
tf.reset_default_graph()
