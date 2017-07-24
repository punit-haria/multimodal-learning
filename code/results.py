import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
import pandas as pd
import numpy as np

cm_choice = cm.Greys  # Greys_r
plt.style.use('ggplot')



def curve_plot(tracker, curve_name, curve_label=None, axis=None, scale_by_batch=True,
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
        parms = trial.parameters

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



def image_plot(tracker, models, data, suffix, n_rows, n_cols,
               n_pixels=300, spacing=0, synthesis_type='reconstruct'):

    n_images = n_rows * n_cols
    assert n_images % 2 == 0
    n = n_images // 2

    for name in tracker.get_runs():

        print("Plotting ", name, flush=True)

        trial = tracker.get(name)
        _model = models[trial.model_name]
        parms = trial.parameters
        parms['n_conditional_pixels'] = n_pixels

        model = initialize(name, _model, parms, data, tracker)
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

        elif synthesis_type == 'latent_activations':
            latent_activation_plot(model, data, 1000, path)

        else:
            raise NotImplementedError

        model.close()


def latent_activation_plot(model, data, n_samples, path):

    x = data.sample(n_samples, dtype='test')
    if type(x) in [list, tuple]:
        x = x[0]

    z = model.encode(x, mean=True)

    z_std = np.std(z, axis=0)

    n_z = len(z_std)

    df = []
    for i in range(n_z):
        row = [z_std[i], "latent_"+str(i)]
        df.append(row)

    df = pd.DataFrame(df, columns=['standard deviation', 'latent variable'])

    sns.set_style("whitegrid")
    ax = sns.barplot(x="latent variable", y="standard deviation", data=df)

    fig = ax.get_figure()
    fig.savefig(path)



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

    images = np.reshape(images, newshape=[n_rows, n_cols]+image_dim)
    images = np.squeeze(images)

    fig, plots = plt.subplots(n_rows, n_cols, figsize=(10,10))

    for i in range(n_rows):
        for j in range(n_cols):
            plots[i,j].imshow(images[i,j], cmap=cm_choice, interpolation='none')
            plots[i,j].axis('off')

    fig.subplots_adjust(wspace=spacing, hspace=spacing)

    plt.savefig(path)
    plt.close('all')


def initialize(name, model, parameters, data, tracker):

    # sample minibatch for weight initialization
    x = data.sample(parameters['batch_size'], dtype='train')
    if type(x) in [list, tuple]:
        x = x[0]

    # constructor
    mod = model(arguments=parameters, name=name, tracker=tracker, init_minibatch=x)

    return mod


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

