import numpy as np
from results import Results, Trial, Series
import plot

import matplotlib.pyplot as plt
plt.style.use('ggplot')


experiment = "experiment_mnist.pickle"
res = Results.load(experiment)


### Figure 1 ###

trials = [
    'joint_vae_Lxy__constrain', 
    'joint_vae_Lxy__share_weights', 
    'joint_vae_Txy_Lx__constrain', 
    'joint_vae_Tyx_Ly__constrain',
    'joint_vae_average__constrain'
]
labels = [
    r'$L_{xy} \ (Constraint)$', 
    r'$L_{xy} \ (Weight \ Sharing)$', 
    r'$T_{xy} + L_x$', 
    r'$T_{yx} + L_y$', 
    r'$Average$'
]

plt.figure(figsize=(12,9))

for i,t in enumerate(trials):
    run = res.get(t)
    steps, series = run.get_series('test_xy_bound')
    plt.plot(steps, series, label=labels[i], linewidth=2)

plt.axis([0,6000,-200,-100])
plt.legend(loc='lower right', fontsize=18)
plt.xlabel('Training Steps (minibatch = 250)')
plt.ylabel('Variational Bound')
#plt.title('MNIST Test Performance')

plt.savefig('../plots/mnist_test.png')
plt.close('all')



### Figure 2 ###

trials = [
    'joint_vae_Lxy__constrain', 
    'joint_vae_Lxy__share_weights', 
    'joint_vae_Txy_Lx__constrain', 
    'joint_vae_Tyx_Ly__constrain',
    'joint_vae_average__constrain'
]

labels = [
    'Lxy_constrain',
    'Lxy_share_weights',
    'Txy_Lx', 'Tyx_Ly', 'average'
]

n_images = 18

for i,t in enumerate(trials):
    run = res.get(t)

    Xb, XX = run.get_series('XtoX', i=10000)
    _, XY = run.get_series('XtoY', i=10000)
    Yb = np.ones(Xb.shape)  * 0.5

    recons = np.concatenate((XX,XY), axis=1)
    origs = np.concatenate((Xb,Yb), axis=1)
    images = np.concatenate((origs[0:n_images],recons[0:n_images]), axis=0)
    images = np.reshape(images, [-1,28,28])

    plot.plot_images(images, 6, 6, '../plots/'+labels[i]+'_reconstruct_from_X.png')

    #----

    Yb, YX = run.get_series('YtoX', i=10000)
    _, YY = run.get_series('YtoY', i=10000)
    Xb = np.ones(Xb.shape)  * 0.5

    recons = np.concatenate((YX,YY), axis=1)
    origs = np.concatenate((Xb,Yb), axis=1)
    images = np.concatenate((origs[0:n_images],recons[0:n_images]), axis=0)
    images = np.reshape(images, [-1,28,28])

    plot.plot_images(images, 6, 6, '../plots/'+labels[i]+'_reconstruct_from_Y.png')

    #----

    Xb, Yb, X, Y = run.get_series('XjYjtoXY', i=10000)

    recons = np.concatenate((X,Y), axis=1)
    origs = np.concatenate((Xb,Yb), axis=1)
    images = np.concatenate((origs[0:n_images],recons[0:n_images]), axis=0)
    images = np.reshape(images, [-1,28,28])

    plot.plot_images(images, 6, 6, '../plots/'+labels[i]+'_reconstruct_from_XandY.png')