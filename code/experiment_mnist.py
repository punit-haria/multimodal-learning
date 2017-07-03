import tensorflow as tf
import numpy as np 

import plot
import utils

from models.joint_vae import JointVAE 
from models.joint_vae_extended import JointVAE_XtoY, JointVAE_YtoX, JointVAE_Average
from models.joint_vae_cnn import JointVAE_CNN 

from data import JointMNIST as MNIST


### PARAMETERS ###

learning_rate = 0.002                           
batch_size = 250  
z_dim = 50                
n_hidden = 200
n_paired = 1000
strategy = 'constrain'

x_dim = 392 
y_dim = 392         
image_dim = (14,28,1)          
train_steps = 10000

# data set
mnist = MNIST(n_paired)

# models
models = {
    'joint_vae_Lxy': JointVAE,
    'joint_vae_Txy_Lx': JointVAE_XtoY,
    'joint_vae_Tyx_Ly': JointVAE_YtoX,
    'joint_vae_average': JointVAE_Average
}

# strategies
strategies = ['share_weights', 'constrain']


for name, model in models.items():
    for strat in strategies:
        
        model_name = name + '__' + strat
        vae = model((x_dim, y_dim), z_dim, learning_rate, n_hidden, strat, name=model_name)

        # train model
        for i in range(train_steps+1):
            
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

                # plot reconstructions 
                if i % 2000 == 0:
                    n_examples = 8

                    Xb = X[0:n_examples]
                    Yb = Y[0:n_examples]
                    XYb = np.concatenate((Xb,Yb), axis=1)

                    XX, YX = vae.reconstruct_from_x(Xb)
                    XY, YY = vae.reconstruct_from_y(Yb)
                    XXY, YXY = vae.reconstruct(Xb,Yb)

                    fromX = np.concatenate((XX,YX), axis=1)
                    fromY = np.concatenate((XY,YY), axis=1)
                    fromXY = np.concatenate((XXY,YXY), axis=1)

                    imagesX = np.concatenate((XYb, fromX), axis=0)
                    imagesY = np.concatenate((XYb, fromY), axis=0)
                    imagesXY = np.concatenate((XYb, fromXY), axis=0)

                    imagesX = np.reshape(imagesX, [-1,28,28])
                    imagesY = np.reshape(imagesY, [-1,28,28])
                    imagesXY = np.reshape(imagesXY, [-1,28,28])

                    plot.plot_images(imagesX, 4, 4, '../plots/'+model_name+'__reconstruct_X_'+str(i))
                    plot.plot_images(imagesY, 4, 4, '../plots/'+model_name+'__reconstruct_Y_'+str(i))
                    plot.plot_images(imagesXY, 4, 4, '../plots/'+model_name+'__reconstruct_XY_'+str(i))


        # save final model
        vae.save_state()

        # reset tensorflow session and graph
        vae.sess.close()
        tf.reset_default_graph()
