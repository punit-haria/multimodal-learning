import tensorflow as tf
import numpy as np 

import utils

from models.joint_vae import JointVAE 
from models.joint_vae_extended import JointVAE_XtoY, JointVAE_YtoX, JointVAE_Average
from models.joint_vae_cnn import JointVAE_CNN 

from data import JointMNIST as MNIST
from results import Results


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
train_steps = 100000
plot_steps = 2500

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


# store experimental results
results = Results('experiment_mnist')


for name, model in models.items():
    for strat in strategies:
        
        # load model
        model_name = name + '__' + strat
        vae = model((x_dim, y_dim), z_dim, learning_rate, n_hidden, strat, name=model_name)

        # store next experimental run
        results.create_run(model_name)

        # train model
        for i in range(train_steps+1):
            
            # random minibatch 
            X, Y, X_joint, Y_joint = mnist.sample('train', batch_size)

            # training step
            x_bound, y_bound, xy_bound = vae.train(X, Y, X_joint, Y_joint)

            # save results
            results.add(i, x_bound, "train_x_bound")
            results.add(i, y_bound, "train_y_bound")
            results.add(i, xy_bound, "train_xy_bound")

            if i % 25 == 0:
                print("At iteration ", i)

                # test minibatch
                X, Y = mnist.sample('test', 1000)

                # test model
                x_bound, y_bound, xy_bound = vae.test(X, Y, X, Y)

                # save results
                results.add(i, x_bound, "test_x_bound")
                results.add(i, y_bound, "test_y_bound")
                results.add(i, xy_bound, "test_xy_bound")

                # plot reconstructions 
                if i % plot_steps == 0:
                    n_examples = 100

                    Xb = X[0:n_examples]
                    Yb = Y[0:n_examples]
                    XYb = np.concatenate((Xb,Yb), axis=1)

                    XX, YX = vae.reconstruct_from_x(Xb)
                    XY, YY = vae.reconstruct_from_y(Yb)
                    XXY, YXY = vae.reconstruct(Xb,Yb)

                    # save reconstructions
                    results.add(i, (Xb, XX), "XtoX")
                    results.add(i, (Xb, YX), "XtoY")
                    results.add(i, (Yb, XY), "YtoX")
                    results.add(i, (Yb, YY), "YtoY")
                    results.add(i, (Xb,Yb,XXY,YXY), "XjYjtoXY")

                    '''
                    fromX = np.concatenate((XX,YX), axis=1)
                    fromY = np.concatenate((XY,YY), axis=1)
                    fromXY = np.concatenate((XXY,YXY), axis=1)

                    imagesX = np.concatenate((XYb, fromX), axis=0)
                    imagesY = np.concatenate((XYb, fromY), axis=0)
                    imagesXY = np.concatenate((XYb, fromXY), axis=0)

                    imagesX = np.reshape(imagesX, [-1,28,28])
                    imagesY = np.reshape(imagesY, [-1,28,28])
                    imagesXY = np.reshape(imagesXY, [-1,28,28])

                    plot.plot_images(imagesX, 6, 6, '../plots/'+model_name+'__reconstruct_X_'+str(i))
                    plot.plot_images(imagesY, 6, 6, '../plots/'+model_name+'__reconstruct_Y_'+str(i))
                    plot.plot_images(imagesXY, 6, 6, '../plots/'+model_name+'__reconstruct_XY_'+str(i))
                    '''

        # save final model
        vae.save_state()

        # reset tensorflow session and graph
        vae.sess.close()
        tf.reset_default_graph()


# save experimental results
Results.save(results)