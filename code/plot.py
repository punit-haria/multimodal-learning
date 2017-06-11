"""
Plotting functions. 
"""
import numpy as np 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cm_choice = cm.Greys

import queue


def plot_images(images, n_rows, n_cols, path):
    """
    Plot images in a grid.
    """
    n = len(images)
    assert n < n_rows*n_cols

    # image queue
    q = queue.Queue()
    for i in range(n):
        q.put(i)

    # figure
    fig, plots = plt.subplots(n_rows, n_cols, figsize=(10,10))

    for i in range(n_cols):
        for j in range(n_rows):
            if q.qsize() == 0:
                break

            # plot next image
            idx = q.get() 
            plots[j,i].imshow(images[idx], cmap=cm_choice, interpolation='none')
            plots[j,i].axis('off') 

    # save figure
    plt.savefig(path)
	plt.cla()





