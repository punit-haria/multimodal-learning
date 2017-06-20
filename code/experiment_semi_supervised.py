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
ytr = utils.one_hot_encoding(ytr, 10)
yte = utils.one_hot_encoding(yte, 10)

# index labelled training examples
n_labelled = 1000
labelled = np.arange(n_labelled)


# pretrained Model M1 
vae = cp.VariationalAutoEncoder(x_dim, z1_dim, learning_rate, 'vae')
vae.load_state(name='vae')  # using final model state

# get embedding space means (Z1)
Ztr = vae.encode(Xtr)
Zte = vae.encode(Xte)

# Model M2
m2 = cp.M2(z1_dim, z2_dim, n_classes, learning_rate, 'm2', session=vae.sess)


# train model
for i in range(train_steps):
    
    # randomly sampled minibatch 
    idx, (Zb, yb) = data.sample([Ztr,ytr], batch_size)

    # separate missing and labelled indices
    missing_vals = np.array(sorted(set(idx) - set(labelled)))
    idx_missing = np.array([np.argwhere(idx == x)[0,0]  for x in missing_vals], dtype=np.int32)
    labelled_vals = np.array(sorted(set(idx) - set(idx_missing)))
    idx_labelled = np.array([np.argwhere(idx == x)[0,0]  for x in labelled_vals], dtype=np.int32)

    # training step
    m2.train(Zb, yb, idx_missing, idx_labelled)

    if i % 100 == 0:
        print("At iteration ", i)

        # test minibatch
        _, (Ztb, ytb) = data.sample([Zte,yte], 1000)

        # test model
        accuracy = m2.test(Ztb, ytb)

        # test classification accuracy
        print("Test Accuracy: ", accuracy)

    if i % 2000 == 0:        
        m2.save_state(name='M2_'+str(i))  # save current model

# save final model
m2.save_state()
