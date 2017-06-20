import tensorflow as tf
import numpy as np 

import components as cp
import data
import plot
import utils


# parameters
learning_rate = 0.001                            
batch_size = 100  
n_classes = 10
x_dim = 784                    
z1_dim = 20                        
z2_dim = 10
train_steps = 1000                      


# train/test sets
Xtr, ytr, Xte, yte = data.mnist()

# one hot encoding 


# separate labelled/unlabelled training examples
n_labelled = 1000
Xlab = Xtr[0:n_labelled]
Ylab = ytr[0:n_labelled]
Xmiss = Xtr[n_labelled:]
Ymiss = Ytr[n_labelled:]

# pretrained Model M1 
vae = cp.VariationalAutoEncoder(x_dim, z1_dim, learning_rate, 'vae')
vae.load_state(name='vae_60000')

# Model M2
m2 = cp.M2(z1_dim, z2_dim, n_classes, learning_rate, 'm2')

# train model
for i in range(train_steps):
    
    # randomly sampled minibatch 
    Xb = data.sample(Xtr, batch_size)

    # training step
    vae.train(Xb)

    if i % 100 == 0:
        print("At iteration ", i)

        # test minibatch
        Xtb, ytb = data.sample([Xte,yte], 1000)

        # test model
        vae.test(Xtb)

    if i % 2000 == 0:
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

# save final model
vae.save_state()
