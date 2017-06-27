import tensorflow as tf
import numpy as np 

import components as cp
import data
import plot
import utils

from models.joint_vae import JointVAE


# parameters
learning_rate = 0.001                            
batch_size = 100  
x_dim = 392 
y_dim = 392                   
z_dim = 50                    
train_steps = 100 


# joint/missing split
x_and_y = set(np.arange(1000))
x_only = set(len(x_and_y) + np.arange(29500))
y_only = set(len(x_and_y) + len(x_only) + np.arange(29500))


# train/test sets
Xtr, ytr, Xte, yte = data.mnist()

# joint variational auto-encoder
vae = JointVAE((x_dim, y_dim), z_dim, learning_rate)


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

'''
    if i % 1000 == 0:
        print("At iteration ", i)

        # test minibatch
        idx, Xtb = data.sample(Xte, 1000)

        # test model
        vae.test(X, Y, X_joint, Y_joint)

    if i % 1000 == 0:
        if z_dim == 2:
            # plot decoded images from uniform grid in latent space
            n_grid = 7
            Z = utils.generate_uniform(n_grid,3)
            images = np.reshape(vae.decode(Z), [-1,28,28])
            plot.plot_images(images, n_grid, n_grid, '../plots/images_'+str(i))

            # plot latent space
            Zmean = vae.encode(Xtb)
            plot.plot_latent_space(Zmean, ytb, '../plots/latent_'+str(i))

        # plot reconstruction samples
        Xtb = Xtb[0:8]
        Xrec = vae.reconstruct(Xtb)
        images = np.concatenate((Xrec, Xtb), axis=0)
        images = np.reshape(images, [-1,28,28])
        plot.plot_images(images, 4, 4, '../plots/reconstructions_'+str(i))
        
        # save current model
        vae.save_state(name='vae_'+str(i))
'''

# save final model
vae.save_state()
