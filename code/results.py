import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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



def image_plot(tracker, models, data, n_rows, n_cols, syntheses,
               n_pixels=300, spacing=0, suffix=None, model_type='regular'):

    for name in tracker.get_runs():

        print("Plotting ", name, flush=True)

        trial = tracker.get(name)
        _model = models[trial.model_name]
        parms = trial.parameters
        parms['n_conditional_pixels'] = n_pixels

        if suffix is None:
            suffix = str(parms['train_steps'])

        model = initialize(name, _model, parms, data, tracker, model_type)
        model.load_state(suffix=suffix)

        for synthesis_type in syntheses:

            print("Synthesis: ", synthesis_type, flush=True)

            path = '../plots/' + tracker.name + '_' + name.replace(".","-") + '_' + synthesis_type

            if synthesis_type == 'reconstruct':
                reconstruction(model, data, parms, spacing, n_rows, n_cols, model_type, path)

            elif synthesis_type == 'fix_latents':
                images = fix_latents(model, data, n_rows, n_cols, model_type)
                _image_plot(images, parms, spacing, path)

            elif synthesis_type == 'sample':
                x, rx = separate_samples(model, data, n_rows, n_cols, model_type)
                _image_plot(x, parms, spacing, path+'__test')
                _image_plot(rx, parms, spacing, path+'__model')

            elif synthesis_type == 'latent_activations':
                latent_activation_plot(model, data, 1000, path, model_type)

            else:
                raise NotImplementedError

        model.close()


def initialize(name, model, parameters, data, tracker, model_type):

    # sample minibatch for weight initialization
    x = data.sample(parameters['batch_size'], dtype='train')
    if type(x) in [list, tuple]:
        x = x[0]

    if model_type == 'joint':
        paired = parameters['n_paired_samples']
        unpaired = parameters['n_unpaired_samples']
        xs = sample(data, n_samples=(paired, unpaired), model_type=model_type, dtype='train')

        mod = model(arguments=parameters, name=name, tracker=tracker, init_minibatches=xs)

    else:
        n = parameters['batch_size']
        x = sample(data, n_samples=n, model_type=model_type, dtype='train')

        mod = model(arguments=parameters, name=name, tracker=tracker, init_minibatch=x)

    return mod


def sample(data, n_samples, model_type, dtype='test'):

    if model_type == 'joint':
        if dtype == 'test':
            x1, x2 = data.sample_stratified(n_paired_samples=n_samples, dtype='test')
            return x1, x2

        else:
            paired, unpaired = n_samples
            x1, x2, x1p, x2p = data.sample_stratified(n_paired_samples=paired, n_unpaired_samples=unpaired,
                                                      dtype='train')
            return x1, x2, x1p, x2p

    else:
        x = data.sample(n_samples, dtype=dtype)
        if type(x) in [list, tuple]:
            x = x[0]

        return x


def latent_activation_plot(model, data, n_samples, path):

    if model.is_flow:
        print("CAN'T PLOT LATENT ACTIVATIONS FOR NORMALIZING FLOW!")
        return

    import seaborn as sns

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



def reconstruction(model, data, parms, spacing, n_rows, n_cols, model_type, path):

    n_x = model.n_x

    n_images = n_rows * n_cols
    assert n_cols % 2 == 0
    n = n_images // 2

    if model_type == 'joint':

        names = ['joint', 'joint', 'translate', 'translate', 'marginal', 'marginal']
        ims = []

        x1, x2 = sample(data, n_samples=n, model_type=model_type, dtype='test')
        rx1, rx2 = model.reconstruct((x1, x2))
        ims.append((x1, rx1))
        ims.append((x2, rx2))

        x1, x2 = sample(data, n_samples=n, model_type=model_type, dtype='test')
        _, rx2 = model.reconstruct((x1, None))
        rx1, _ = model.reconstruct((None, x2))
        ims.append((x1, rx2))
        ims.append((x2, rx1))

        x1, x2 = sample(data, n_samples=n, model_type=model_type, dtype='test')
        rx1, _ = model.reconstruct((x1, None))
        _, rx2 = model.reconstruct((None, x2))
        ims.append((x1, rx1))
        ims.append((x2, rx2))

        for name, tup in zip(names, ims):
            x, rx = tup

            images = np.empty((n_images, n_x))
            images[0::2] = x
            images[1::2] = rx

            images = np.reshape(images, newshape=[n_rows, n_cols, n_x])

            current_path = path + "_" + name
            _image_plot(images, parms, spacing, current_path)

    else:
        x = sample(data, n_samples=n, model_type=model_type, dtype='test')

        rx = model.reconstruct(x)

        images = np.empty((n_images, n_x))
        images[0::2] = x
        images[1::2] = rx

        images = np.reshape(images, newshape=[n_rows, n_cols, n_x])

        _image_plot(images, parms, spacing, path)



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
        for j in range(n_cols):
            if j == 0:
                images[i,j,:] = x[i]
            else:
                images[i,j,:] = rxs[j-1][i]

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


def plot_latent_space(Z, Y, path):
    """
    Plots latent space, colored with labels. Assumes latent space is 2-dimensional.

    Z: latent values
    Y: labels
    path: save figure to this path
    """
    import seaborn as sns

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

