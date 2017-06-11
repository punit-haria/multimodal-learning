"""
Plotting functions. 
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cm_choice = cm.Greys  # Greys_r

import seaborn as sns
import pandas as pd
import numpy as np 
import queue


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
    fig.map(plt.scatter, 'z1', 'z2').add_legend()
    
    # save
    plt.savefig(path)
    plt.close()

