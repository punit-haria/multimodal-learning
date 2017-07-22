import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
import pandas as pd
import numpy as np

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



def image_plot(tracker, models, parms, data, suffix, n_rows, n_cols,
               spacing=0, synthesis_type='reconstruct'):

    names = tracker.get_runs()

    n_images = n_rows * n_cols
    assert n_images % 2 == 0
    n = n_images // 2

    for name in names:
        trial = tracker.get(name)
        _model = models[trial.model_name]

        model = _model(arguments=parms, name=name, tracker=None)
        model.load_state(suffix=suffix)

        path = '../plots/' + tracker.name + '_' + name + '_' + synthesis_type

        if synthesis_type == 'reconstruct':
            images = reconstruction(model, data, n_rows, n_cols)
            _image_plot(images, parms, spacing, path)

        elif synthesis_type == 'fix_latents':
            images = fix_latents(model, data, n_rows, n_cols)
            _image_plot(images, parms, spacing, path)

        elif synthesis_type == 'sample':
            x, rx = separate_samples(model, data, n_rows, n_cols)
            _image_plot(x, parms, spacing, path+'__test')
            _image_plot(rx, parms, spacing, path+'__model')

        else:
            raise NotImplementedError

        model.close()



def reconstruction(model, data, n_rows, n_cols):

    n_images = n_rows * n_cols

    assert n_cols % 2 == 0

    n = n_images // 2

    x = data.sample(n, dtype='test')
    if type(x) in [list, tuple]:
        x = x[0]

    n_x = x.shape[1]

    rx = model.reconstruct(x)

    images = np.empty((n_images, n_x))
    images[0::2] = x
    images[1::2] = rx

    images = np.reshape(images, newshape=[n_rows, n_cols, n_x])
    images = np.transpose(images, axes=[1,0,2])

    return images



def separate_samples(model, data, n_rows, n_cols):

    n = n_rows * n_cols

    x = data.sample(n, dtype='test')
    if type(x) in [list, tuple]:
        x = x[0]

    n_x = x.shape[1]

    z = model.sample_prior(n)
    rx = model.decode(z)

    x = np.reshape(x, newshape=[n_rows, n_cols, n_x])
    rx = np.reshape(rx, newshape=[n_rows, n_cols, n_x])

    return x, rx



def fix_latents(model, data, n_rows, n_cols):

    n_vary = n_cols - 1

    x = data.sample(n_rows, dtype='test')
    if type(x) in [list, tuple]:
        x = x[0]

    n_x = x.shape[1]

    z = model.encode(x, mean=False)

    rxs = []
    for i in range(n_vary):
        rx = model.decode(z)
        rxs.append(rx)

    images = np.empty((n_rows, n_cols, n_x))

    for i in range(n_rows):
        for j in range(n_vary):
            if i == 0:
                images[i,j,:] = x[i]
            else:
                images[i,j,:] = rxs[j][i]

    return images




def _image_plot(images, parms, spacing, path):

    h = parms['height']
    w = parms['width']
    n_ch = parms['n_channels']
    image_dim = [h, w, n_ch]

    n_rows = images.shape[0]
    n_cols = images.shape[1]

    np.reshape(images, newshape=[n_rows, n_cols]+image_dim)

    fig, plots = plt.subplots(n_rows, n_cols, figsize=(10,10))

    for i in range(n_rows):
        for j in range(n_cols):
            plots[i,j].imshow(images[i,j], cmap=cm_choice, interpolation='none')
            plots[i,j].axis('off')

    fig.subplots_adjust(wspace=spacing, hspace=spacing)

    plt.savefig(path)
    plt.close('all')



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

