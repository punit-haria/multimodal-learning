import tensorflow as tf
import numpy as np 

import components as cp
import data
import plot
import utils


# parameters
learning_rate = 0.001                            
batch_size = 100  
x_dim = 784                    
z_dim = 2                        
train_steps = 5000                        


# train/test sets
Xtr, ytr, Xte, yte = data.mnist()

# variational auto-encoder (model + optimization)
vae = cp.VariationalAutoEncoder(x_dim, z_dim, learning_rate, 'vae')


# train model
for i in range(train_steps):
    
    # randomly sampled minibatch 
    Xb = data.sample(Xtr, batch_size)

    # training step
    vae.train(Xb)

    if i % 250 == 0:
        print("At iteration ", i)

        # test model
        vae.test(Xte)
        
        # plot decoded images from uniform grid in latent space
        Z = utils.generate_uniform(20,3)
        images = np.reshape(vae.decode(Z), [-1,28,28])
        plot.plot_images(images, 20, 20, '../plots/images_'+str(i))

        # plot latent space
        Zmean = vae.encode(Xte)
        plot.plot_latent_space(Zmean, yte, '../plots/latent_'+str(i))

        # plot reconstruction samples
        Xtb = data.sample(Xte, 8)
        Xrec = vae.reconstruct(Xtb)
        images = np.concatenate((Xrec, Xtb), axis=0)
        images = np.reshape(images, [-1,28,28])
        plot.plot_images(images, 4, 4, '../plots/reconstructions_'+str(i))


# save final model
vae.save_state()
