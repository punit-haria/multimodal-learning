import numpy as np
from results import Results
import plot

import matplotlib.pyplot as plt
plt.style.use('ggplot')


### PARAMETERS ###

experiment = "experiment_vae_conditional_pixelcnn"

res = Results.load(experiment+'.pickle')

test_axis = [0,res.i,460,550]  # axis range for test lower bound
train_axis = [0,res.i,460,550]  # axis range for train lower bound
image_dim = [28,28]


### Figure 1 ###

runs = res.get_runs()
labels = [_x + ' Lower Bound' for _x in runs]

plt.figure(figsize=(12,9))

for i,t in enumerate(runs):
    run = res.get(t)
    steps, series = run.get_series('test_loss')  # test_lower_bound
    plt.plot(steps, series, label=labels[i], linewidth=2)

plt.axis(test_axis)
plt.legend(loc='lower right', fontsize=18)
plt.xlabel('Steps')
plt.ylabel('Log-Likelihood Lower Bound')

plt.savefig('../plots/'+experiment+'_test_bound.png')
plt.close('all')



### Figure 2 ###

runs = res.get_runs()
labels = [_x + ' Lower Bound' for _x in runs]

plt.figure(figsize=(12,9))

for i,t in enumerate(runs):
    run = res.get(t)
    steps, series = run.get_series('train_loss')  # train_lower_bound
    plt.plot(steps, series, label=labels[i], linewidth=2)

plt.axis(train_axis)
plt.legend(loc='lower right', fontsize=18)
plt.xlabel('Steps')
plt.ylabel('Log-Likelihood Lower Bound')

plt.savefig('../plots/'+experiment+'_train_bound.png')
plt.close('all')



### Figure 3 ###

n_images = 18
labels = res.get_runs()

for i,t in enumerate(runs):
    print("At trial: ", t, flush=True)

    run = res.get(t)

    series_names = ['x_reconstructed']

    for ser in series_names:

        steps, series = run.get_series(ser)

        for time_step, value in zip(steps, series):
            print("Timestep: ", time_step, flush=True)

            xb, rx_probs = value

            images = np.concatenate([xb, rx_probs], axis=0)
            images = np.reshape(images, [-1] + image_dim)

            plot.plot_images(images, 6, 6, '../plots/'+labels[i]+'_'+ser+'_'+str(time_step)+'.png')

