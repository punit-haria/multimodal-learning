import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
import numpy as np

cm_choice = cm.Greys  # Greys_r
plt.style.use('ggplot')

'''

Helper methods and classes for synthesizing images under various conditions. 

'''


def coco_plot(tracker, models, data, n_rows, n_cols, train_steps=None, repetitions=1):

    for name in tracker.get_runs():

        print("Plotting ", name, flush=True)

        trial = tracker.get(name)
        _model = models[trial.model_name]
        parms = trial.parameters

        if train_steps is None:
            train_steps = str(parms['train_steps'])

        model = _coco_initialize(name, _model, parms, data, tracker)
        model.load_state(suffix=train_steps)

        print("Translation (with mean)", flush=True)
        path = '../plots/' + name.replace(".", "-") + '_withMean_' + train_steps
        for cc in range(repetitions):
            path_ext = path + '_' + str(cc)
            _coco_reconstruct(model, data, parms, n_rows, n_cols, mean=True, path=path_ext)


        print("Translation (stochastic)", flush=True)
        path = '../plots/' + name.replace(".", "-") + '_stochastic_' + train_steps
        for cc in range(repetitions):
            path_ext = path + '_' + str(cc)
            _coco_reconstruct(model, data, parms, n_rows, n_cols, mean=False, path=path_ext)

        model.close()


def _coco_initialize(name, model, parameters, data, tracker):

    paired = parameters['n_paired_samples']
    unpaired = parameters['n_unpaired_samples']

    xs = data.sample_stratified(n_paired_samples=paired, n_unpaired_samples=unpaired, dtype='train')

    mod = model(arguments=parameters, name=name, tracker=tracker, init_minibatches=xs)

    return mod


def _coco_reconstruct(model, data, parms, n_rows, n_cols, mean, path):

    n = n_rows * n_cols

    xi, xc = data.sample_stratified(n_paired_samples=n, dtype='test')

    _, rxc = model.reconstruct((xi, None), mean=mean)
    rxi, _ = model.reconstruct((None, xc), mean=mean)

    xc = _get_caption_text(data, xc)
    rxc = _get_caption_text(data, rxc)

    # rxi: float ndarray --> batch_size x (48*64*3)
    # rxc: grid of strings --> n_rows x n_cols

    # images to captions
    _coco_image_plot(xi, rxc, n_rows, n_cols, path=path+'_translate_images')

    # captions to images
    _coco_image_plot(rxi, xc, n_rows, n_cols, path=path + '_translate_captions')



def _coco_image_plot(images, capts, n_rows, n_cols, path):

    images = np.reshape(images, newshape=[n_rows, n_cols, 48, 64, 3])

    fig, plots = plt.subplots(n_rows, n_cols, figsize=(10,10))

    for i in range(n_rows):
        for j in range(n_cols):
            plots[i,j].imshow(images[i,j], cmap=cm_choice, interpolation='none')
            plots[i,j].axis('off')
            plots[i,j].set_title(capts[i][j])

    fig.subplots_adjust(wspace=5, hspace=5)

    plt.savefig(path)
    plt.close('all')


def _get_caption_text(data, capts, n_rows, n_cols):

    # capts: n x max_seq_len
    capts = np.reshape(capts, newshape=[n_rows, n_cols, -1])
    # capts: n_rows x n_cols x max_seq_len

    captions = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            c = []
            seq = capts[i,j,:]
            for w in seq:
                if w == data._padding:
                    break

                word = data.get_word(w)
                c.append(word)

            ' '.join(c)
            row.append(c)
        captions.append(row)

    return captions






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
               n_pixels=300, spacing=0, suffix=None, model_type='regular', count=1):

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

            path = '../plots/' + name.replace(".","-") + '_' + synthesis_type + '_' + suffix

            for cc in range(count):

                path_ext = path + '_'+str(cc)

                if synthesis_type == 'reconstruct':
                    reconstruction(model, data, parms, spacing, n_rows, n_cols, model_type, path_ext)

                elif synthesis_type == 'fix_latents':
                    fix_latents(model, data, parms, spacing, n_rows, n_cols, model_type, path_ext)

                elif synthesis_type == 'repeated_synth':
                    if model_type == "joint":
                        repeat_reconstruct(model, data, parms, spacing, n_rows, n_cols, model_type, path_ext)
                    else:
                        print("WARNING: Not implemented for regular model.")

                elif synthesis_type == 'sample':
                    separate_samples(model, data, parms, spacing, n_rows, n_cols, model_type, path_ext)

                elif synthesis_type == 'latent_activations':
                    if model_type == "joint":
                        print("WARNING: Not implemented for joint model.")
                    else:
                        latent_activation_plot(model, data, 1000, path_ext)

                else:
                    raise NotImplementedError

        model.close()


def initialize(name, model, parameters, data, tracker, model_type):

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

        if parms['data'] == 'halved_mnist':

            names = ['translate_x1', 'translate_x2', 'joint']

            ims = []

            x1, x2 = sample(data, n_samples=n, model_type=model_type, dtype='test')

            rx1_1, rx2_1 = model.reconstruct((x1, None))
            rx1_2, rx2_2 = model.reconstruct((None, x2))
            rx1f, rx2f = model.reconstruct((x1, x2))

            image_dim = [14, 28, 1]
            x1 = np.reshape(x1, newshape=[-1]+image_dim)
            x2 = np.reshape(x2, newshape=[-1]+image_dim)
            rx1_1 = np.reshape(rx1_1, newshape=[-1] + image_dim)
            rx2_1 = np.reshape(rx2_1, newshape=[-1] + image_dim)
            rx1_2 = np.reshape(rx1_2, newshape=[-1] + image_dim)
            rx2_2 = np.reshape(rx2_2, newshape=[-1] + image_dim)
            rx1f = np.reshape(rx1f, newshape=[-1] + image_dim)
            rx2f = np.reshape(rx2f, newshape=[-1] + image_dim)

            x1_mask = np.copy(x1.astype(float))
            x2_mask = np.copy(x2.astype(float))
            x1_mask[x1_mask == 0] = 0.5
            x2_mask[x2_mask == 0] = 0.5
            x1_mask[x1_mask == 1] = 0.8
            x2_mask[x2_mask == 1] = 0.8

            x1_full = np.concatenate((x1, x2_mask), axis=1)
            x2_full = np.concatenate((x1_mask, x2), axis=1)
            x = np.concatenate((x1, x2), axis=1)

            x1_full = np.reshape(x1_full, newshape=[n, -1])
            x2_full = np.reshape(x2_full, newshape=[n, -1])
            x = np.reshape(x, newshape=[n, -1])

            t1 = np.concatenate((rx1_1, rx2_1), axis=1)
            t2 = np.concatenate((rx1_2, rx2_2), axis=1)
            tf = np.concatenate((rx1f, rx2f), axis=1)

            t1 = np.reshape(t1, newshape=[n, -1])
            t2 = np.reshape(t2, newshape=[n, -1])
            tf = np.reshape(tf, newshape=[n, -1])

            ims.append((x1_full, t1))
            ims.append((x2_full, t2))
            ims.append((x, tf))

            for name, tup in zip(names, ims):

                x, rx = tup

                images = np.empty((n_images, n_x*2))
                images[0::2] = x
                images[1::2] = rx

                images = np.reshape(images, newshape=[n_rows, n_cols, n_x*2])

                current_path = path + "_" + name
                _image_plot(images, parms, spacing, current_path, h=28, w=28)


        else:
            #names = ['joint_x1', 'joint_x2', 'translate_x1', 'translate_x2', 'marginal_x1', 'marginal_x2']
            names = ['translate_x1', 'translate_x2']
            ims = []

            #x1, x2 = sample(data, n_samples=n, model_type=model_type, dtype='test')
            #rx1, rx2 = model.reconstruct((x1, x2))
            #ims.append((x1, rx1))
            #ims.append((x2, rx2))

            x1, x2 = sample(data, n_samples=n, model_type=model_type, dtype='test')
            _, rx2 = model.reconstruct((x1, None))
            rx1, _ = model.reconstruct((None, x2))
            ims.append((x1, rx2))
            ims.append((x2, rx1))

            #x1, x2 = sample(data, n_samples=n, model_type=model_type, dtype='test')
            #rx1, _ = model.reconstruct((x1, None))
            #_, rx2 = model.reconstruct((None, x2))
            #ims.append((x1, rx1))
            #ims.append((x2, rx2))

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



def separate_samples(model, data, parms, spacing, n_rows, n_cols, model_type, path):

    n_x = model.n_x

    n = n_rows * n_cols

    if model_type == "joint":
        if parms['data'] == 'halved_mnist':

            x1, x2 = sample(data, n_samples=n, model_type=model_type, dtype='test')

            image_dim = [14, 28, 1]
            x1 = np.reshape(x1, newshape=[n_rows, n_cols] + image_dim)
            x2 = np.reshape(x2, newshape=[n_rows, n_cols] + image_dim)

            x = np.concatenate((x1, x2), axis=2)
            x = np.reshape(x, newshape=[n_rows, n_cols, n_x*2])

            _image_plot(x, parms, spacing, path + '__testset', h=28, w=28)

            z = model.sample_prior(n)
            rx1, rx2 = model.decode(z)

            rx1 = np.reshape(rx1, newshape=[n_rows, n_cols] + image_dim)
            rx2 = np.reshape(rx2, newshape=[n_rows, n_cols] + image_dim)

            rx = np.concatenate((rx1, rx2), axis=2)
            rx = np.reshape(rx, newshape=[n_rows, n_cols, n_x*2])

            _image_plot(rx, parms, spacing, path + '__model', h=28, w=28)

        else:

            x1, x2 = sample(data, n_samples=n, model_type=model_type, dtype='test')

            x1 = np.reshape(x1, newshape=[n_rows, n_cols, n_x])
            x2 = np.reshape(x2, newshape=[n_rows, n_cols, n_x])

            _image_plot(x1, parms, spacing, path + '__testset_x1')
            _image_plot(x2, parms, spacing, path + '__testset_x2')

            z = model.sample_prior(n//2)
            rx1, rx2 = model.decode(z)

            images = np.empty((n, n_x))
            images[0::2] = rx1
            images[1::2] = rx2

            images = np.reshape(images, newshape=[n_rows, n_cols, n_x])

            _image_plot(images, parms, spacing, path + '__model')

            #rx1 = np.reshape(rx1, newshape=[n_rows, n_cols, n_x])
            #rx2 = np.reshape(rx2, newshape=[n_rows, n_cols, n_x])

            #_image_plot(rx1, parms, spacing, path + '__model_x1')
            #_image_plot(rx2, parms, spacing, path + '__model_x2')

    else:
        x = sample(data, n_samples=n, model_type=model_type, dtype='test')

        z = model.sample_prior(n)
        rx = model.decode(z)

        x = np.reshape(x, newshape=[n_rows, n_cols, n_x])
        rx = np.reshape(rx, newshape=[n_rows, n_cols, n_x])

        _image_plot(x, parms, spacing, path + '__testset')
        _image_plot(rx, parms, spacing, path + '__model')


def fix_latents(model, data, parms, spacing, n_rows, n_cols, model_type, path):

    if model_type == "joint":

        if parms['data'] == 'halved_mnist':
            print("WARNING: Not implemented for halved MNIST dataset.")

        else:
            n_vary = n_cols - 1
            x1, x2 = sample(data, n_samples=n_rows, model_type=model_type, dtype='test')

            z1 = model.encode((x1, None), mean=False)
            rx2s = []
            for i in range(n_vary):
                _, rx2 = model.decode(z1)
                rx2s.append(rx2)

            z2 = model.encode((None, x2), mean=False)
            rx1s = []
            for i in range(n_vary):
                rx1, _ = model.decode(z2)
                rx1s.append(rx1)

            z = model.encode((x1, x2), mean=False)
            rxs = []
            for i in range(n_vary):
                rx1, rx2 = model.decode(z)
                rxs.append((rx1,rx2))

            n_x = x1.shape[1]
            n_x2 = x2.shape[1]
            assert n_x == n_x2

            ims1 = np.empty((n_rows, n_cols, n_x))
            ims2 = np.empty((n_rows, n_cols, n_x))
            ims = np.empty((n_rows, n_cols, n_x*2))

            h = parms['height']
            w = parms['width']
            n_ch = parms['n_channels']

            x1_im = np.reshape(x1, newshape=[n_rows, h, w, n_ch])
            x2_im = np.reshape(x2, newshape=[n_rows, h, w, n_ch])
            xj = np.concatenate((x1_im, x2_im), axis=2)
            xj = np.reshape(xj, newshape=[n_rows, n_x * 2])

            for j in range(n_cols):

                rx1_im = np.reshape(rxs[j-1][0], newshape=[n_rows, h, w, n_ch])
                rx2_im = np.reshape(rxs[j-1][1], newshape=[n_rows, h, w, n_ch])
                rxj = np.concatenate((rx1_im, rx2_im), axis=2)
                rxj = np.reshape(rxj, newshape=[n_rows, n_x * 2])

                for i in range(n_rows):
                    if j == 0:
                        ims1[i, j, :] = x1[i]
                        ims2[i, j, :] = x2[i]
                        ims[i, j, :] = xj[i]

                    else:
                        ims1[i, j, :] = rx2s[j - 1][i]
                        ims2[i, j, :] = rx1s[j - 1][i]
                        ims[i, j, :] = rxj[i]

            names = ['joint', 'translate_x1', 'translate_x2']

            _image_plot(ims1, parms, spacing, path+'_translate_x1')
            _image_plot(ims2, parms, spacing, path+'_translate_x2')
            _image_plot(ims, parms, spacing, path+'_joint', h=h, w=w*2)

    else:
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

        _image_plot(images, parms, spacing, path)


def repeat_reconstruct(model, data, parms, spacing, n_rows, n_cols, model_type, path):

    if model_type == "joint":
        if parms['data'] == 'halved_mnist':
            print("WARNING: Not implemented for halved MNIST dataset.")

        else:
            n_vary = n_cols - 1
            x1, x2 = sample(data, n_samples=n_rows, model_type=model_type, dtype='test')

            rx2s = []
            for i in range(n_vary):
                _, rx2 = model.reconstruct((x1, None))
                rx2s.append(rx2)

            rx1s = []
            for i in range(n_vary):
                rx1, _ = model.reconstruct((None, x2))
                rx1s.append(rx1)

            rxs = []
            for i in range(n_vary):
                rx1, rx2 = model.reconstruct((x1, x2))
                rxs.append((rx1,rx2))

            n_x = x1.shape[1]
            n_x2 = x2.shape[1]
            assert n_x == n_x2

            ims1 = np.empty((n_rows, n_cols, n_x))
            ims2 = np.empty((n_rows, n_cols, n_x))
            ims = np.empty((n_rows, n_cols, n_x*2))

            h = parms['height']
            w = parms['width']
            n_ch = parms['n_channels']

            x1_im = np.reshape(x1, newshape=[n_rows, h, w, n_ch])
            x2_im = np.reshape(x2, newshape=[n_rows, h, w, n_ch])
            xj = np.concatenate((x1_im, x2_im), axis=2)
            xj = np.reshape(xj, newshape=[n_rows, n_x * 2])

            for j in range(n_cols):

                rx1_im = np.reshape(rxs[j-1][0], newshape=[n_rows, h, w, n_ch])
                rx2_im = np.reshape(rxs[j-1][1], newshape=[n_rows, h, w, n_ch])
                rxj = np.concatenate((rx1_im, rx2_im), axis=2)
                rxj = np.reshape(rxj, newshape=[n_rows, n_x * 2])

                for i in range(n_rows):
                    if j == 0:
                        ims1[i, j, :] = x1[i]
                        ims2[i, j, :] = x2[i]
                        ims[i, j, :] = xj[i]

                    else:
                        ims1[i, j, :] = rx2s[j - 1][i]
                        ims2[i, j, :] = rx1s[j - 1][i]
                        ims[i, j, :] = rxj[i]

            names = ['joint', 'translate_x1', 'translate_x2']

            _image_plot(ims1, parms, spacing, path+'_translate_x1')
            _image_plot(ims2, parms, spacing, path+'_translate_x2')
            _image_plot(ims, parms, spacing, path+'_joint', h=h, w=w*2)




def _image_plot(images, parms, spacing, path, h=None, w=None):

    if h is None:
        h = parms['height']
    if w is None:
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

