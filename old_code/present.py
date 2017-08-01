import numpy as np
from results import Results, Trial, Series
import plot

import matplotlib.pyplot as plt
plt.style.use('ggplot')


experiment = "experiment_mnist_cnn.pickle"
res = Results.load(experiment)


### Figure 1 ###

trials = ['vae_joint', 'vae_translate', 'vae_cnn']
labels = ['Joint Bound', 'Translation Bound', 'CNN Joint Bound']

plt.figure(figsize=(12,9))

for i,t in enumerate(trials):
    if res.contains(t):
        run = res.get(t)
        steps, series = run.get_series('test_lower_bound')
        plt.plot(steps, series, label=labels[i], linewidth=2)

plt.axis([0,5000,-5000,-100])
plt.legend(loc='lower right', fontsize=18)
plt.xlabel('Training Steps (minibatch = 250)')
plt.ylabel('Log-Likelihood Lower Bound')
plt.title('MNIST Test Performance')

plt.savefig('../plots/mnist_test.png')
plt.close('all')



### Figure 2 ###

n_images = 18
time_steps = [500, 1000, 2000]

for time_step in time_steps:
    
    print("Timestep: ", time_step, flush=True)

    for i,t in enumerate(trials):
        print("At trial: ", t, flush=True)

        if not res.contains(t):
            continue

        run = res.get(t)

        x1b, rx1_1 = run.get_series('x1_1', i=time_step)
        _, rx2_1 = run.get_series('x2_1', i=time_step)
        x2b = np.ones(x1b.shape)  * 0.5

        recons = np.concatenate((rx1_1,rx2_1), axis=1)
        origs = np.concatenate((x1b,x2b), axis=1)
        images = np.concatenate((origs[0:n_images],recons[0:n_images]), axis=0)
        images = np.reshape(images, [-1,28,28])

        plot.plot_images(images, 6, 6, '../plots/'+labels[i]+'_reconstruct_from_x1_'+str(time_step)+'.png')

        #----

        x2b, rx1_2 = run.get_series('x1_2', i=time_step)
        _, rx2_2 = run.get_series('x2_2', i=time_step)
        x1b = np.ones(x2b.shape)  * 0.5

        recons = np.concatenate((rx1_2,rx2_2), axis=1)
        origs = np.concatenate((x1b,x2b), axis=1)
        images = np.concatenate((origs[0:n_images],recons[0:n_images]), axis=0)
        images = np.reshape(images, [-1,28,28])

        plot.plot_images(images, 6, 6, '../plots/'+labels[i]+'_reconstruct_from_x2_'+str(time_step)+'.png')

        #----

        x1b, x2b, rx1p, rx2p = run.get_series('x12p', i=time_step)

        recons = np.concatenate((rx1p,rx2p), axis=1)
        origs = np.concatenate((x1b,x2b), axis=1)
        images = np.concatenate((origs[0:n_images],recons[0:n_images]), axis=0)
        images = np.reshape(images, [-1,28,28])

        plot.plot_images(images, 6, 6, '../plots/'+labels[i]+'_reconstruct_from_both_'+str(time_step)+'.png')

