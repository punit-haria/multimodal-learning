import tensorflow as tf
import numpy as np 

import plot
import utils

from models.joint_vae import JointVAE 
from models.joint_vae_extended import JointVAE_XtoY, JointVAE_YtoX, JointVAE_Average, JointVAE_Deeper
from models.joint_vae_cnn import JointVAE_CNN 

from data import ColouredMNIST as MNIST

print("Starting experiment...", flush=True)

### PARAMETERS ###

learning_rate = 0.002                           
batch_size = 250  
z_dim = 50                
n_hidden = 200
n_paired = 1000
strategy = 'constrain'

x_dim = 784 * 3
y_dim = 784 * 3     
image_dim = (28,28,3)          
train_steps = 50000

# data set
print("Reading data...", flush=True)
mnist = MNIST(n_paired)

# models
models = {
#   'joint_vae_Lxy': JointVAE,
#    'joint_vae_Txy_Lx': JointVAE_XtoY,
#    'joint_vae_Tyx_Ly': JointVAE_YtoX,
#   'joint_vae_average': JointVAE_Average,
#    'joint_vae_cnn': JointVAE_CNN
'joint_vae_Lxy' : JointVAE_Deeper
}


for name, model in models.items():
    name = 'mnist_colored_'+name

    print("Loading model...", flush=True)
    vae = model((x_dim, y_dim), z_dim, learning_rate, n_hidden, strategy, name=name)
    #vae = model((x_dim, y_dim), z_dim, image_dim, learning_rate, n_hidden, strategy, name=name)

    # train model
    print("Training...", flush=True)
    for i in range(train_steps+1):
        
        # random minibatch 
        X, Y, X_joint, Y_joint = mnist.sample('train', batch_size)

        # training step
        vae.train(X, Y, X_joint, Y_joint)

        if i % 25 == 0:
            print("At iteration ", i, flush=True)

            # test minibatch
            X, Y = mnist.sample('test', 1000)

            # test model
            vae.test(X, Y, X, Y)

            # plot reconstructions 
            if i % 500 == 0:
                n_examples = 18

                Xb = X[0:n_examples]
                Yb = Y[0:n_examples]

                YX = vae.translate_x(Xb)
                XY = vae.translate_y(Yb)

                XtoY = np.concatenate((Xb,YX), axis=0)
                YtoX = np.concatenate((Yb,XY), axis=0)

                XtoY = np.reshape(XtoY, [-1,28,28,3])
                YtoX = np.reshape(YtoX, [-1,28,28,3])

                plot.plot_images(XtoY, 6, 6, '../plots/'+name+'__translate_x_'+str(i))
                plot.plot_images(YtoX, 6, 6, '../plots/'+name+'__translate_y_'+str(i))


    # save final model
    vae.save_state()

    # reset tensorflow session and graph
    vae.sess.close()
    tf.reset_default_graph()
