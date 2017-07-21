import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
import pandas as pd
import numpy as np 
import queue

cm_choice = cm.Greys  # Greys_r
plt.style.use('ggplot')



def curve_plot(tracker, parms, curve_name, curve_label=None, axis=None, scale_by_batch=True,
               legend='lower right', legend_font=18, xlab='x', ylab='f(x)'):

    names = tracker.get_runs()

    if curve_label is None:
        labels = names
    else:
        labels = [_x + ' ' + curve_label for _x in names]

    plt.figure(figsize=(12, 9))

    for label, name in zip(labels, names):
        trial = tracker.get(name)
        x, f = trial.get_series(curve_name)

        if scale_by_batch:
            bs = parms['batch_size']
            x = [_x * bs for _x in x]

        plt.plot(x, f, label=label, linewidth=2)

    if axis is not None:
        plt.axis(axis)

    plt.legend(loc=legend, fontsize=legend_font)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.savefig('../plots/' + tracker.name + '_' + curve_name + '.png')
    plt.close('all')



def image_plot(tracker, parms, ):

    h = parms['height']
    w = parms['width']
    n_ch = parms['n_channels']

    image_dim = [h, w, n_ch]




def plot_images(images, n_rows, n_cols, path):
    """
    Plot images in a grid.

    images: matrix of images
    n_rows: number of rows in image grid
    n_cols: number of columns in image grid
    path: save figure to this path
    """
    n = len(images)
    assert n <= n_rows*n_cols

    # image queue
    q = queue.Queue()
    for i in range(n):
        q.put(i)

    # figure
    fig, plots = plt.subplots(n_rows, n_cols, figsize=(10,10))

    for i in range(n_cols):
        for j in range(n_rows):
            # plot next image in queue
            if q.qsize() > 0:
                idx = q.get() 
                plots[j,i].imshow(images[idx], cmap=cm_choice, interpolation='none')
            plots[j,i].axis('off') 

    # save figure
    plt.savefig(path)
    plt.close()



def plot_latent_space(Z, Y, path):
    """
    Plots latent space, colored with labels. Assumes latent space is 2-dimensional.

    Z: latent values
    Y: labels
    path: save figure to this path
    """
    # create dataframe for plotting
    Y = np.expand_dims(Y, axis=1)
    X = np.concatenate((Z,Y), axis=1)
    df = pd.DataFrame(X)
    df.columns = ['z1','z2','digit']
    df.digit = df.digit.astype('object')

    # plot with seaborn 
    fig = sns.FacetGrid(data=df, hue='digit', aspect=1, size=6)
    fig.map(plt.scatter, 'z1', 'z2')
    
    # save
    plt.savefig(path)
    plt.close()

