import tensorflow as tf
import numpy as np
from models import sequential as sq


def seq_encoder(x, vocab_size, embed_size, n_units, n_z, n_layers, init, scope):
    """
    RNN encoder using GRUs.
    """
    with tf.variable_scope(scope):

        nonlin = tf.nn.elu

        if init:
            embeddings = tf.get_variable("embeddings", shape=[vocab_size, embed_size],
                                         initializer=tf.random_normal_initializer(0, 0.05))
        else:
            embeddings = tf.get_variable("embeddings", shape=[vocab_size, embed_size])

        x = tf.nn.embedding_lookup(embeddings, x)   # batch_size x max_seq_len x embed_size

        gru = sq.GRUCell(num_units=n_units, activation=nonlin, init=init)
        gru = tf.nn.rnn_cell.MultiRNNCell([gru] * n_layers)
        out, state = tf.nn.dynamic_rnn(gru, x, dtype=tf.float32, initial_state=None)

        mu = linear(out, n_out=n_z, init=init, scope="mu_layer")   # assuming 'out' is n_z dimensional vector h_t

        sigma = linear(x, n_z, init=init, scope="sigma_layer")
        sigma = tf.nn.softplus(sigma)

        return mu, sigma, out, x



def seq_decoder(z, x, embed_size, nxc, n_layers, init, scope):
    """
    RNN decoder using GRUs.
    """
    with tf.variable_scope(scope):

        nonlin = tf.nn.elu

        z = linear(z, n_out=embed_size, init=init, scope='mu_sigma_layer')  # batch_size x embed_size

        # x: batch_size x max_seq_len x embed_size
        max_seq_len = x.get_shape()[1].value
        x = tf.slice(x, begin=[0,0,0], size=[-1,max_seq_len-1,-1])

        z = tf.concat([z,x], axis=1)

        gru = sq.GRUCell(num_units=nxc, activation=nonlin, init=init)
        gru = tf.nn.rnn_cell.MultiRNNCell([gru] * n_layers)
        out, state = tf.nn.dynamic_rnn(gru, z, dtype=tf.float32, initial_state=None)

        return out



def convolution_coco(x, nch, n_fmaps, n_units, n_z, init, scope):
    """
    Convolutional network for images of dimension 48 x 64 x nch.
    """
    with tf.variable_scope(scope):

        x = tf.reshape(x, shape=[-1, 48, 64, nch])
        nonlin = tf.nn.elu

        x = conv_residual_block(x, k=3, n_feature_maps=n_fmaps, nonlinearity=nonlin,
                                stride=True, init=init, scope='res_1')
        x = nonlin(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_fmaps, nonlinearity=nonlin,
                                stride=True, init=init, scope='res_2')
        x = nonlin(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_fmaps, nonlinearity=nonlin,
                                stride=False, init=init, scope='unstrided_1')
        x = nonlin(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_fmaps, nonlinearity=nonlin,
                                stride=True, init=init, scope='res_3')
        x = nonlin(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_fmaps, nonlinearity=nonlin,
                                stride=True, init=init, scope='res_4')
        x = nonlin(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_fmaps, nonlinearity=nonlin,
                                stride=False, init=init, scope='unstrided_2')
        x = nonlin(x)

        x = tf.contrib.layers.flatten(x)

        x = linear(x, n_out=n_units, init=init, scope='linear_layer')
        x = nonlin(x)

        mu = linear(x, n_z, init=init, scope="mu_layer")

        sigma = linear(x, n_z, init=init, scope="sigma_layer")
        sigma = tf.nn.softplus(sigma)

        return mu, sigma, x


def deconvolution_coco(z, nch, n_fmaps, n_units, init, scope):
    """
    Deconvolution network for images of dimension 48 x 64 x nch.
    """
    with tf.variable_scope(scope):

        nonlin = tf.nn.elu

        z = linear(z, n_out=n_units, init=init, scope='mu_sigma_layer')
        z = nonlin(z)

        h = 3
        w = 4
        dim = h * w * n_fmaps
        z = linear(z, n_out=dim, init=init, scope='linear_layer')
        z = nonlin(z)

        z = tf.reshape(z, shape=[-1, h, w, n_fmaps])


        z = deconv_residual_block(z, k=3, n_feature_maps=n_fmaps, out_ch=n_fmaps,
                                  nonlinearity=nonlin, stride=False, init=init, scope='unstrided_2')
        z = nonlin(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_fmaps, out_ch=n_fmaps,
                                  nonlinearity=nonlin, stride=True, init=init, scope='res_4')
        z = nonlin(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_fmaps, out_ch=n_fmaps,
                                  nonlinearity=nonlin, stride=True, init=init, scope='res_3')
        z = nonlin(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_fmaps, out_ch=n_fmaps,
                                  nonlinearity=nonlin, stride=False, init=init, scope='unstrided_1')
        z = nonlin(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_fmaps, out_ch=n_fmaps, nonlinearity=nonlin,
                                  stride=True, init=init, scope='res_2')
        z = nonlin(z)


        z = deconv_residual_block(z, k=3, n_feature_maps=n_fmaps, out_ch=nch, nonlinearity=nonlin,
                                  stride=True, init=init, scope='res_1')

        return z


def joint_coco_encode(h1, h2, n_units, n_z, init, scope):

    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu

        h1 = linear(h1, n_units, init=init, scope='layer_h1')
        h2 = linear(h2, n_units, init=init, scope='layer_h2')

        h12 = nonlinearity(h1 + h2)

        mean = linear(h12, n_z, init=init, scope="mean_layer")

        sigma = linear(h12, n_z, init=init, scope="var_layer")
        sigma = tf.nn.softplus(sigma)

        return mean, sigma





def convolution_daynight(x, n_ch, n_feature_maps, n_units, n_z, extra, init, scope):
    """
    Convolution network for use with DayNight dataset.
    """
    with tf.variable_scope(scope):

        x = tf.reshape(x, shape=[-1, 44, 64, n_ch])
        nonlinearity = tf.nn.elu

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_1')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_2')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=False, init=init, scope='unstrided_1')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_3')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_4')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=False, init=init, scope='unstrided_2')
        x = nonlinearity(x)


        x = tf.contrib.layers.flatten(x)

        x = linear(x, n_out=n_units, init=init, scope='linear_layer')
        x = nonlinearity(x)

        mu = linear(x, n_z, init=init, scope="mu_layer")

        sigma = linear(x, n_z, init=init, scope="sigma_layer")
        sigma = tf.nn.softplus(sigma)

        h = linear(x, n_z, init=init, scope="h_layer") if extra else None

        return mu, sigma, h, x



def deconvolution_daynight(z, n_ch, n_feature_maps, n_units, init, scope):
    """
    Deconvolution network for use with DayNight dataset.
    """
    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu

        z = linear(z, n_out=n_units, init=init, scope='mu_sigma_layer')
        z = nonlinearity(z)

        h = 2
        w = 4
        dim = h * w * n_feature_maps
        z = linear(z, n_out=dim, init=init, scope='linear_layer')
        z = nonlinearity(z)

        z = tf.reshape(z, shape=[-1, h, w, n_feature_maps])


        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=False, init=init, scope='unstrided_2')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_4')
        z = tf.pad(z, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_3')
        z = tf.pad(z, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=False, init=init, scope='unstrided_1')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_2')
        z = nonlinearity(z)


        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_1')

        return z




def convolution_sketchy(x, n_ch, n_feature_maps, n_units, n_z, extra, init, scope):
    """
    Convolution network for use with Sketchy dataset.
    """
    with tf.variable_scope(scope):

        x = tf.reshape(x, shape=[-1, 64, 64, n_ch])
        nonlinearity = tf.nn.elu

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_1')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_2')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=False, init=init, scope='unstrided_1')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_3')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_4')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=False, init=init, scope='unstrided_2')
        x = nonlinearity(x)


        x = tf.contrib.layers.flatten(x)

        x = linear(x, n_out=n_units, init=init, scope='linear_layer')
        x = nonlinearity(x)

        mu = linear(x, n_z, init=init, scope="mu_layer")

        sigma = linear(x, n_z, init=init, scope="sigma_layer")
        sigma = tf.nn.softplus(sigma)

        h = linear(x, n_z, init=init, scope="h_layer") if extra else None

        return mu, sigma, h, x



def deconvolution_sketchy(z, n_ch, n_feature_maps, n_units, init, scope):
    """
    Deconvolution network for use with Sketchy dataset.
    """
    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu

        z = linear(z, n_out=n_units, init=init, scope='mu_sigma_layer')
        z = nonlinearity(z)

        h = w = 4
        dim = h * w * n_feature_maps
        z = linear(z, n_out=dim, init=init, scope='linear_layer')
        z = nonlinearity(z)

        z = tf.reshape(z, shape=[-1, h, w, n_feature_maps])


        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=False, init=init, scope='unstrided_2')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_4')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_3')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=False, init=init, scope='unstrided_1')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_2')
        z = nonlinearity(z)


        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_1')

        return z




def convolution_cifar(x, n_ch, n_feature_maps, n_units, n_z, extra, init, scope):
    """
    Convolution network for use with CIFAR dataset.
    """
    with tf.variable_scope(scope):

        x = tf.reshape(x, shape=[-1, 32, 32, n_ch])
        nonlinearity = tf.nn.elu

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_1')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=False, init=init, scope='unstrided_1')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_2')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_3')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=False, init=init, scope='unstrided_2')
        x = nonlinearity(x)


        x = tf.contrib.layers.flatten(x)

        x = linear(x, n_out=n_units, init=init, scope='linear_layer')
        x = nonlinearity(x)

        mu = linear(x, n_z, init=init, scope="mu_layer")

        sigma = linear(x, n_z, init=init, scope="sigma_layer")
        sigma = tf.nn.softplus(sigma)

        h = linear(x, n_z, init=init, scope="h_layer") if extra else None

        return mu, sigma, h, x


def deconvolution_cifar(z, n_ch, n_feature_maps, n_units, init, scope):
    """
    Deconvolution network for use with CIFAR dataset.
    """
    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu

        z = linear(z, n_out=n_units, init=init, scope='mu_sigma_layer')
        z = nonlinearity(z)

        h = w = 4
        dim = h * w * n_feature_maps
        z = linear(z, n_out=dim, init=init, scope='linear_layer')
        z = nonlinearity(z)

        z = tf.reshape(z, shape=[-1, h, w, n_feature_maps])


        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=False, init=init, scope='unstrided_2')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_3')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_2')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=False, init=init, scope='unstrided_1')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_1')

        return z


def convolution_mnist(x, n_ch, n_feature_maps, n_units, n_z, extra, init, scope):
    """
    Convolution network for use with MNIST dataset.
    """
    with tf.variable_scope(scope):

        x = tf.reshape(x, shape=[-1, 28, 28, n_ch])
        nonlinearity = tf.nn.elu

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_1')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=False, init=init, scope='unstrided_1')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_2')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_3')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=False, init=init, scope='unstrided_2')
        x = nonlinearity(x)  #

        x = tf.contrib.layers.flatten(x)

        x = linear(x, n_out=n_units, init=init, scope='linear_layer')
        x = nonlinearity(x)

        mu = linear(x, n_z, init=init, scope="mu_layer")

        sigma = linear(x, n_z, init=init, scope="sigma_layer")
        sigma = tf.nn.softplus(sigma)

        h = linear(x, n_z, init=init, scope="h_layer") if extra else None

        return mu, sigma, h, x


def deconvolution_mnist(z, n_ch, n_feature_maps, n_units, init, scope):
    """
    Deconvolution network for use with MNIST dataset.
    """
    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu

        z = linear(z, n_out=n_units, init=init, scope='mu_sigma_layer')
        z = nonlinearity(z)

        h = w = 3
        dim = h * w * n_feature_maps
        z = linear(z, n_out=dim, init=init, scope='linear_layer')
        z = nonlinearity(z)

        z = tf.reshape(z, shape=[-1, h, w, n_feature_maps])

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=False, init=init, scope='unstrided_2')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_3')
        z = tf.pad(z, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]])
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_2')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=False, init=init, scope='unstrided_1')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_1')

        return z


def convolution_halved_mnist(x, n_ch, n_feature_maps, n_units, n_z, extra, init, scope):
    """
    Convolution network for use with halved MNIST dataset.
    """
    with tf.variable_scope(scope):

        x = tf.reshape(x, shape=[-1, 14, 28, n_ch])
        nonlinearity = tf.nn.elu

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_1')
        x = nonlinearity(x)

        #x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
        #                        stride=False, init=init, scope='unstrided_1')
        #x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_2')
        x = nonlinearity(x)

        #x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
        #                        stride=False, init=init, scope='unstrided_2')
        #x = nonlinearity(x)

        x = tf.contrib.layers.flatten(x)

        x = linear(x, n_out=n_units, init=init, scope='linear_layer')
        x = nonlinearity(x)

        mu = linear(x, n_z, init=init, scope="mu_layer")

        sigma = linear(x, n_z, init=init, scope="sigma_layer")
        sigma = tf.nn.softplus(sigma)

        h = linear(x, n_z, init=init, scope="h_layer") if extra else None

        return mu, sigma, h, x


def deconvolution_halved_mnist(z, n_ch, n_feature_maps, n_units, init, scope):
    """
    Deconvolution network for use with MNIST dataset.
    """
    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu

        z = linear(z, n_out=n_units, init=init, scope='mu_sigma_layer')
        z = nonlinearity(z)

        h = 3
        w = 7
        dim = h * w * n_feature_maps
        z = linear(z, n_out=dim, init=init, scope='linear_layer')
        z = nonlinearity(z)

        z = tf.reshape(z, shape=[-1, h, w, n_feature_maps])

        #z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
        #                          nonlinearity=nonlinearity, stride=False, init=init, scope='unstrided_2')
        #z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_2')
        z = tf.pad(z, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
        z = nonlinearity(z)

        #z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
        #                          nonlinearity=nonlinearity, stride=False, init=init, scope='unstrided_1')
        #z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_1')

        return z



def conv_residual_block(c, k, n_feature_maps, nonlinearity, stride, init, scope):
    """
    Residual Block.
    """
    with tf.variable_scope(scope):

        id = c

        c = conv(c, k=k, out_ch=n_feature_maps, stride=False, init=init, scope='layer_1')
        c = nonlinearity(c)
        c = conv(c, k=k, out_ch=n_feature_maps, stride=stride, init=init, scope='layer_2')

        if stride:
            id = conv(id, k=k, out_ch=n_feature_maps, stride=True, init=init, scope='identity_downsampled')

        c = c + id

        return c


def deconv_residual_block(d, k, n_feature_maps, out_ch, nonlinearity, stride, init, scope):
    """
    Deconvolution residual block.
    """
    with tf.variable_scope(scope):

        id = d

        d = deconv(d, k=k, out_ch=n_feature_maps, stride=stride, init=init, scope='layer_1')
        d = nonlinearity(d)
        d = deconv(d, k=k, out_ch=out_ch, stride=False, init=init, scope='layer_2')

        if stride:
            id = deconv(id, k=k, out_ch=out_ch, stride=True, init=init, scope='identity_upsampled')

        d = d + id

        return d


def conv(x, k, out_ch, stride, init, scope, mask_type=None):
    """
    Convolution layer
    """
    with tf.variable_scope(scope):

        in_ch = x.get_shape()[3].value

        strides = [1, 2, 2, 1] if stride else [1, 1, 1, 1]
        w_shape = [k, k, in_ch, out_ch]

        if init:
            v = tf.get_variable("v", shape=w_shape, initializer=tf.random_normal_initializer(0,0.05))
            v = v.initialized_value()

            if mask_type is not None:
                mask = conv_mask(w_shape, mask_type)
                v = tf.multiply(v, tf.constant(mask))

            v_norm = tf.nn.l2_normalize(v, dim = [0, 1, 2])

            t = tf.nn.conv2d(x, v_norm, strides=strides, padding='SAME')
            mu_t, var_t = tf.nn.moments(t, axes=[0,1,2])

            inv = 1 / tf.sqrt(var_t + 1e-10)
            _ = tf.get_variable("g", initializer=inv)
            _ = tf.get_variable("b", initializer=-mu_t * inv)

            inv = tf.reshape(inv, shape=[1, 1, 1, out_ch])
            mu_t = tf.reshape(mu_t, shape=[1, 1, 1, out_ch])

            return tf.multiply(t - mu_t, inv)

        else:
            v = tf.get_variable("v", shape=w_shape)
            g = tf.get_variable("g", shape=[out_ch])
            b = tf.get_variable("b", shape=[out_ch])

            if mask_type is not None:
                mask = conv_mask(w_shape, mask_type)
                v = tf.multiply(v, tf.constant(mask))

            w = tf.reshape(g, shape=[1,1,1,out_ch]) * tf.nn.l2_normalize(v, dim=[0,1,2])
            b = tf.reshape(b, shape=[1,1,1,out_ch])

            return tf.nn.conv2d(x, w, strides=strides, padding='SAME') + b


def deconv(x, k, out_ch, stride, init, scope):
    """
    Deconvolution layer (transpose of convolution)
    """
    with tf.variable_scope(scope):

        batch_size = tf.shape(x)[0]
        height = x.get_shape()[1].value
        width = x.get_shape()[2].value
        in_ch = x.get_shape()[3].value

        if stride:
            out_shape = [batch_size, height*2, width*2, out_ch]
            strides = [1, 2, 2, 1]
        else:
            out_shape = [batch_size, height, width, out_ch]
            strides = [1, 1, 1, 1]

        w_shape = [k, k, out_ch, in_ch]

        if init:
            v = tf.get_variable("v", shape=w_shape, initializer=tf.random_normal_initializer(0,0.05))
            v_norm = tf.nn.l2_normalize(v.initialized_value(), dim=[0,1,3])

            t = tf.nn.conv2d_transpose(x, v_norm, output_shape=out_shape, strides=strides, padding='SAME')
            out_shape[0] = None
            t.set_shape(out_shape)

            mu_t, var_t = tf.nn.moments(t, axes=[0,1,2])

            inv = 1 / tf.sqrt(var_t + 1e-10)
            _ = tf.get_variable("g", initializer=inv)
            _ = tf.get_variable("b", initializer=-mu_t * inv)

            inv = tf.reshape(inv, shape=[1, 1, 1, out_ch])
            mu_t = tf.reshape(mu_t, shape=[1, 1, 1, out_ch])

            return tf.multiply(t - mu_t, inv)

        else:
            v = tf.get_variable("v", shape=w_shape)
            g = tf.get_variable("g", shape=[out_ch])
            b = tf.get_variable("b", shape=[out_ch])

            w = tf.reshape(g, shape=[1,1,out_ch,1]) * tf.nn.l2_normalize(v, dim=[0,1,3])
            b = tf.reshape(b, shape=[1,1,1,out_ch])

            d = tf.nn.conv2d_transpose(x, w, output_shape=out_shape, strides=strides, padding='SAME')
            out_shape[0] = None
            d.set_shape(out_shape)

            return d + b


def conv_mask(m_shape, mask_type):

    k = m_shape[0]  # filter size
    assert k == m_shape[1]
    assert k % 2 == 1  # check that k is odd

    if k == 1:
        assert mask_type == 'B'
        return np.ones(shape=m_shape, dtype=np.float32)

    else:
        mask = np.zeros(shape=m_shape, dtype=np.float32)
        half_k = (k // 2) + 1
        for i in range(half_k):
            for j in range(k):
                if i < half_k - 1 or j < half_k - 1:
                    mask[i, j, :, :] = 1

        # mask type
        if mask_type == 'A':
            mask[half_k - 1, half_k - 1, :, :] = 0
        elif mask_type == 'B':
            mask[half_k - 1, half_k - 1, :, :] = 1
        else:
            raise Exception("Masking type not implemented..")

        return mask


def pool(x, scope, reuse):
    """
    Max pooling layer (reduces size by 2)
    """
    with tf.variable_scope(scope, reuse=reuse):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def pixel_cnn(x, n_layers, ka, kb, out_ch, n_feature_maps, init, scope):
    """
    PixelCNN network based on architectures specified in:
        https://arxiv.org/abs/1601.06759
        https://arxiv.org/abs/1701.05517
    ka/kb: widths of mask A and B convolution filters
    """
    with tf.variable_scope(scope):

        n_ch = n_feature_maps
        nonlinearity = tf.nn.elu

        c = conv(x, k=ka, out_ch=n_ch, stride=False, mask_type='A', init=init, scope='layer_1')

        for i in range(n_layers):
            name  = 'residual_block_' + str(i+2)
            c = masked_residual_block(c, kb, nonlinearity, init=init, scope=name)

        c = nonlinearity(c)
        c = conv(c, k=1, out_ch=n_ch, stride=False, mask_type='B', init=init, scope='final_1x1_a')
        c = nonlinearity(c)
        c = conv(c, k=1, out_ch=out_ch, stride=False, mask_type='B', init=init, scope='final_1x1_b')

        return c


def deconvolution_sketchy_ar(x, z, out_ch, n_feature_maps, n_units, n_ar_layers, init, scope):
    """
    Autoregressive deconvolution network for use with Sketchy dataset.
    """
    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu
        n_ch = n_feature_maps

        z = linear(z, n_out=n_units, init=init, scope='mu_sigma_layer')
        z = nonlinearity(z)

        h = w = 4
        dim = h * w * n_feature_maps
        z = linear(z, n_out=dim, init=init, scope='linear_layer')
        z = nonlinearity(z)

        z = tf.reshape(z, shape=[-1, h, w, n_feature_maps])


        z = deconv_residual_block(z, k=3, n_feature_maps=n_ch, out_ch=n_ch,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_4')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_ch, out_ch=n_ch,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_3')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_ch, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_2')
        z = nonlinearity(z)


        z = deconv_residual_block(z, k=3, n_feature_maps=n_ch, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_1')
        z = nonlinearity(z)

        ka = 3
        kb = 3

        # AR layers:

        c = conv(x, k=ka, out_ch=n_ch, stride=False, mask_type='A', init=init, scope='layer_1x')

        for i in range(n_ar_layers):
            cz = conv(z, k=3, out_ch=n_ch, stride=False, mask_type=None, init=init, scope='cond_z_' + str(i+2))
            c = c + cz

            #c = c + z
            c = masked_residual_block(c, kb, nonlinearity, init=init, scope='resblock_' + str(i+2))

        c = nonlinearity(c)

        c = conv(c, k=1, out_ch=n_ch, stride=False, mask_type='B', init=init, scope='final_1x1_a')
        c = nonlinearity(c)
        c = conv(c, k=1, out_ch=out_ch, stride=False, mask_type='B', init=init, scope='final_1x1_b')

        return c





def deconvolution_cifar_ar(x, z, out_ch, n_feature_maps, n_units, n_ar_layers, init, scope):
    """
    Deconvolution network for use with CIFAR dataset.
    """
    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu
        n_ch = n_feature_maps

        z = linear(z, n_out=n_units, init=init, scope='mu_sigma_layer')
        z = nonlinearity(z)

        h = w = 4
        dim = h * w * n_ch
        z = linear(z, n_out=dim, init=init, scope='linear_layer')
        z = nonlinearity(z)

        z = tf.reshape(z, shape=[-1, h, w, n_ch])


        z = deconv_residual_block(z, k=3, n_feature_maps=n_ch, out_ch=n_ch,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_3')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_ch, out_ch=n_ch,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_2')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_ch, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_1')
        z = nonlinearity(z)

        ka = 3
        kb = 3

        # AR layers:

        c = conv(x, k=ka, out_ch=n_ch, stride=False, mask_type='A', init=init, scope='layer_1x')

        for i in range(n_ar_layers):
            cz = conv(z, k=3, out_ch=n_ch, stride=False, mask_type=None, init=init, scope='cond_z_' + str(i+2))
            c = c + cz

            #c = c + z
            c = masked_residual_block(c, kb, nonlinearity, init=init, scope='resblock_' + str(i+2))

        c = nonlinearity(c)

        c = conv(c, k=1, out_ch=n_ch, stride=False, mask_type='B', init=init, scope='final_1x1_a')
        c = nonlinearity(c)
        c = conv(c, k=1, out_ch=out_ch, stride=False, mask_type='B', init=init, scope='final_1x1_b')

        return c



def deconvolution_mnist_ar(x, z, out_ch, n_feature_maps, n_units, n_ar_layers, init, scope):
    """
    Autoregressive decoder for use with MNIST dataset.
    """
    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu
        n_ch = n_feature_maps

        z = linear(z, n_out=n_units, init=init, scope='mu_sigma_layer')
        z = nonlinearity(z)

        h = w = 3
        dim = h * w * n_ch
        z = linear(z, n_out=dim, init=init, scope='linear_layer')
        z = nonlinearity(z)

        z = tf.reshape(z, shape=[-1, h, w, n_ch])

        z = deconv_residual_block(z, k=3, n_feature_maps=n_ch, out_ch=n_ch,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_3')
        z = tf.pad(z, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]])
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_ch, out_ch=n_ch,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_2')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_ch, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_1')
        z = nonlinearity(z)

        ka = 3
        kb = 3

        # AR layers:

        c = conv(x, k=ka, out_ch=n_ch, stride=False, mask_type='A', init=init, scope='layer_1x')

        for i in range(n_ar_layers):
            cz = conv(z, k=3, out_ch=n_ch, stride=False, mask_type=None, init=init, scope='cond_z_' + str(i+2))
            c = c + cz

            #c = c + z
            c = masked_residual_block(c, kb, nonlinearity, init=init, scope='resblock_' + str(i+2))

        c = nonlinearity(c)

        c = conv(c, k=1, out_ch=n_ch, stride=False, mask_type='B', init=init, scope='final_1x1_a')
        c = nonlinearity(c)
        c = conv(c, k=1, out_ch=out_ch, stride=False, mask_type='B', init=init, scope='final_1x1_b')

        return c


def conditional_pixel_cnn_everytime(x, z, n_layers, out_ch, n_feature_maps, init, scope):
    """
    Conditional PixelCNN
    """
    with tf.variable_scope(scope):

        n_ch = n_feature_maps
        nonlinearity = tf.nn.elu
        ka = 3
        kb = 3

        c = conv(x, k=ka, out_ch=n_ch, stride=False, mask_type='A', init=init, scope='layer_1x')

        z = nonlinearity(z)

        for i in range(n_layers):
            cz = conv(z, k=3, out_ch=n_ch, stride=False, mask_type=None, init=init, scope='cond_z_' + str(i+2))
            c = c + cz  # do we need to convolve z??

            c = masked_residual_block(c, kb, nonlinearity, init=init, scope='resblock_' + str(i+2))

        c = nonlinearity(c)

        c = conv(c, k=1, out_ch=n_ch, stride=False, mask_type='B', init=init, scope='final_1x1_a')
        c = nonlinearity(c)
        c = conv(c, k=1, out_ch=out_ch, stride=False, mask_type='B', init=init, scope='final_1x1_b')

        return c


def conditional_pixel_cnn_no_conditionals(x, z, n_layers, out_ch, n_feature_maps, init, scope):
    """
    Conditional PixelCNN
    """
    with tf.variable_scope(scope):

        n_ch = n_feature_maps
        nonlinearity = tf.nn.elu
        ka = 3
        kb = 3

        c = conv(x, k=ka, out_ch=n_ch, stride=False, mask_type='A', init=init, scope='layer_1x')

        c = c + z

        for i in range(n_layers):
            c = masked_residual_block(c, kb, nonlinearity, init=init, scope='resblock_' + str(i+2))

        c = nonlinearity(c)

        c = conv(c, k=1, out_ch=n_ch, stride=False, mask_type='B', init=init, scope='final_1x1_a')
        c = nonlinearity(c)
        c = conv(c, k=1, out_ch=out_ch, stride=False, mask_type='B', init=init, scope='final_1x1_b')

        return c


def masked_residual_block(c, k, nonlinearity, init, scope):
    """
    Residual Block for PixelCNN. See https://arxiv.org/abs/1601.06759
    """
    with tf.variable_scope(scope):

        n_ch = c.get_shape()[3].value
        half_ch = n_ch // 2
        c1 = nonlinearity(c)
        c1 = conv(c1, k=1, out_ch=half_ch, stride=False, mask_type='B', init=init, scope='1x1_a')
        c1 = nonlinearity(c1)
        c1 = conv(c1, k=k, out_ch=half_ch, stride=False, mask_type='B', init=init, scope='conv')
        c1 = nonlinearity(c1)
        c1 = conv(c1, k=1, out_ch=n_ch, stride=False, mask_type='B', init=init, scope='1x1_b')
        c = c1 + c

        return c


def normalizing_flow(mu0, sigma0, h, epsilon, K, n_units, flow_type, init, scope):
    """
    Normalizing flow.
    """
    with tf.variable_scope(scope):

        z = mu0 + tf.multiply(sigma0, epsilon)

        #log_q = -tf.reduce_sum(tf.log(sigma0) + 0.5 * tf.square(epsilon) + 0.5 * np.log(2 * np.pi), axis=1)

        D = z.get_shape()[1].value
        log_q = -tf.log(sigma0) - 0.5*tf.square(epsilon) - 0.5*np.log(2*np.pi)

        for i in range(K):
            if i > 0:
                z = tf.reverse(z, axis=[1])

            if flow_type == "cnn":
                m, s = pixelcnn_flow(z, h=h, init=init, scope='cnn_flow_'+str(i+1))
            else:
                m, s = made_flow(z, h=h, n_units=n_units, init=init, scope='made_flow_'+str(i+1))

            sigma = tf.nn.sigmoid(s)

            z = tf.multiply(sigma, z) + tf.multiply(1-sigma, m)

            #log_q = log_q - tf.reduce_sum(tf.log(sigma), axis=1)
            log_q = log_q - tf.log(sigma)

        return z, log_q



def made_flow(z, h, n_units, init, scope):
    """
    Masked Network (MADE) based on https://arxiv.org/abs/1502.03509
    used as single normalizing flow transform.
    """
    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu
        n_z = z.get_shape()[1].value
        m = None

        h = linear(h, n_out=n_z, init=init, scope='layer_h')
        h = nonlinearity(h)
        z = z + h

        z, m = ar_linear(z, n_out=n_units, m_prev=m, is_final=False, init=init, scope='layer_1')
        z = nonlinearity(z)

        z, m = ar_linear(z, n_out=n_units, m_prev=m, is_final=False, init=init, scope='layer_2')
        z = nonlinearity(z)

        mu, _ = ar_linear(z, n_out=n_z, m_prev=m, is_final=True, init=init, scope='layer_m')
        s, _ = ar_linear(z, n_out=n_z, m_prev=m, is_final=True, is_sigma=True, init=init, scope='layer_s')

        return mu, s


def pixelcnn_flow(z, h, init, scope):

    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu

        z = tf.reshape(z, shape=[-1, 7, 7, 1])
        h = tf.reshape(h, shape=[-1, 7, 7, 1])

        ch = conv(h, k=3, out_ch=3, stride=False, mask_type=None, init=init, scope='layer_1z')
        ch = nonlinearity(ch)

        cz = conv(z, k=3, out_ch=3, stride=False, mask_type='A', init=init, scope='layer_1x')
        c = ch + cz

        c = masked_residual_block(c, k=3, nonlinearity=nonlinearity, init=init, scope='resblock_1')
        c = nonlinearity(c)

        c = conv(c, k=1, out_ch=3, stride=False, mask_type='B', init=init, scope='final_1x1_a')
        c = nonlinearity(c)

        m = conv(c, k=1, out_ch=1, stride=False, mask_type='B', init=init, scope='final_1x1_b_m')
        s = conv(c, k=1, out_ch=1, stride=False, mask_type='B', init=init, scope='final_1x1_b_s')

        m = tf.contrib.layers.flatten(m)
        s = tf.contrib.layers.flatten(s)

        return m, s


def ar_linear(x, n_out, m_prev, is_final, init, scope, is_sigma=False):
    """
    Masked linear transform based on MADE network (https://arxiv.org/abs/1502.03509)
    Results in autoregressive relationship between input and output.
    """
    with tf.variable_scope(scope):

        Kin = x.get_shape()[1].value
        Kout = n_out

        if init:
            v = tf.get_variable("v", shape=[Kin, Kout], initializer=tf.random_normal_initializer(0,0.05))
            v = v.initialized_value()

            v_norm = tf.nn.l2_normalize(v, dim=0)

            t = tf.matmul(x, v_norm)
            mu_t, var_t = tf.nn.moments(t, axes=0)

            inv = 1 / tf.sqrt(var_t + 1e-10)

            _ = tf.get_variable("g", initializer=inv)

            if is_sigma:
                _ = tf.get_variable("b", shape=[Kout], initializer=tf.constant_initializer(1))
            else:
                _ = tf.get_variable("b", initializer=-mu_t * inv)

            inv = tf.reshape(inv, shape=[1, n_out])
            mu_t = tf.reshape(mu_t, shape=[1, n_out])

            return tf.multiply(t - mu_t, inv), None

        else:
            if m_prev is None:
                m_prev = np.arange(Kin) + 1

            if is_final:
                m = np.arange(Kout)
            else:
                m = np.random.randint(low=np.min(m_prev), high=Kin, size=Kout)

            mask = np.zeros(shape=[Kin, Kout], dtype=np.float32)
            for kin in range(Kin):
                for kout in range(Kout):
                    if m[kout] >= m_prev[kin]:
                        mask[kin, kout] = 1

            v = tf.get_variable("v", shape=[Kin, Kout])
            g = tf.get_variable("g", shape=[Kout])
            b = tf.get_variable("b", shape=[Kout])

            scaling = g / tf.sqrt(tf.reduce_sum(tf.square(v), axis=0))

            v = tf.multiply(v, tf.constant(mask))
            x = tf.matmul(x, v)

            scaling = tf.reshape(scaling, shape=[1, n_out])
            b = tf.reshape(b, shape=[1, n_out])

            return tf.multiply(scaling, x) + b, m


def ar_linear_no_init(x, n_out, m_prev, is_final, init, scope):
    """
    Masked linear transform based on MADE network (https://arxiv.org/abs/1502.03509)
    Results in autoregressive relationship between input and output.
    """
    with tf.variable_scope(scope):

        Kin = x.get_shape()[1].value
        Kout = n_out

        if init:
            if m_prev is None:
                assert is_final == False
                m_prev = np.arange(Kin) + 1
                m = np.random.randint(low=1, high=Kin, size=Kout)
            elif is_final:
                m = np.arange(Kout)
            else:
                m = np.random.randint(low=np.min(m_prev), high=Kin, size=Kout)

            mask = np.zeros(shape=[Kin, Kout], dtype=np.float32)
            for kin in range(Kin):
                for kout in range(Kout):
                    if m[kout] >= m_prev[kin]:
                        mask[kin, kout] = 1

            mask = tf.get_variable("mask", initializer=tf.constant(mask), trainable=False)
            mask = mask.initialized_value()

            w = tf.get_variable("w", shape=[Kin, Kout], initializer=tf.contrib.layers.xavier_initializer())
            w = w.initialized_value()
            b = tf.get_variable("b", shape=[Kout], initializer=tf.constant_initializer(0.1))
            b = b.initialized_value()

            w = tf.multiply(w, mask)
            return tf.matmul(x, w) + b, m

        else:
            w = tf.get_variable("w", shape=[Kin, Kout])
            b = tf.get_variable("b", shape=[Kout])

            mask = tf.get_variable("mask", shape=[Kin, Kout], trainable=False)

            w = tf.multiply(w, mask)
            return tf.matmul(x, w) + b, None


def ar_mult(x, n_out, init, scope):
    """
    Matrix multiplication with simple lower triangular mask.
    Results in autoregressive relationship between input and output.
    (Component in Masked Autoencoder)
    """
    with tf.variable_scope(scope):
        n_in = x.get_shape()[1].value

        if init:
            w = tf.get_variable("w", shape=[n_in, n_out], initializer=tf.contrib.layers.xavier_initializer())
            w = w.initialized_value()
        else:
            w = tf.get_variable("w", shape=[n_in, n_out])

        # strictly upper triangular mask
        mask = np.ones(shape=[n_in, n_out], dtype=np.float32)
        mask = np.triu(mask)

        w = tf.multiply(w, tf.constant(mask))
        x = tf.matmul(x, w)

        return x



def joint_fc_encode(h1, h2, n_units, n_z, extra, init, scope):

    with tf.variable_scope(scope):

        nonlinearity = tf.nn.elu

        h1 = linear(h1, n_units, init=init, scope='layer_h1')
        h2 = linear(h2, n_units, init=init, scope='layer_h2')

        h12 = nonlinearity(h1 + h2)

        mean = linear(h12, n_z, init=init, scope="mean_layer")

        sigma = linear(h12, n_z, init=init, scope="var_layer")
        sigma = tf.nn.softplus(sigma)

        h = linear(h12, n_z, init=init, scope="h_layer") if extra else None

        return mean, sigma, h



def fc_encode(x, n_units, n_z, extra, init, scope):
    """
    2 layer fully-connected encoder, as in AEVB paper.
    """
    with tf.variable_scope(scope):
        nonlinearity = tf.nn.elu

        x = linear(x, n_units, init=init, scope="layer_1")
        x = nonlinearity(x)

        x = linear(x, n_units, init=init, scope="layer_2")
        x = nonlinearity(x)

        mean = linear(x, n_z, init=init, scope="mean_layer")

        sigma = linear(x, n_z, init=init, scope="var_layer")
        sigma = tf.nn.softplus(sigma)

        h = linear(x, n_z, init=init, scope="h_layer") if extra else None

        return mean, sigma, h, x


def fc_decode(z, n_units, n_x, init, scope):
    """
    2 layer fully-connected decoder, as in AEVB paper.
    """
    with tf.variable_scope(scope):
        nonlinearity = tf.nn.elu

        z = linear(z, n_units, init=init, scope="layer_1")
        z = nonlinearity(z)

        z = linear(z, n_units, init=init, scope="layer_2")
        z = nonlinearity(z)

        z = linear(z, n_x, init=init, scope="logits_layer")

        return z


def linear(x, n_out, init, scope):
    """
    Linear tranform
    """
    with tf.variable_scope(scope):

        n_x = x.get_shape()[1].value

        if init:
            v = tf.get_variable("v", shape=[n_x, n_out], initializer=tf.random_normal_initializer(0,0.05))
            v_norm = tf.nn.l2_normalize(v.initialized_value(), dim=0)

            t = tf.matmul(x, v_norm)
            mu_t, var_t = tf.nn.moments(t, axes=0)

            inv = 1 / tf.sqrt(var_t + 1e-10)
            _ = tf.get_variable("g", initializer=inv)
            _ = tf.get_variable("b", initializer=-mu_t * inv)

            inv = tf.reshape(inv, shape=[1, n_out])
            mu_t = tf.reshape(mu_t, shape=[1, n_out])

            return tf.multiply(t - mu_t, inv)

        else:
            v = tf.get_variable("v", shape=[n_x, n_out])
            g = tf.get_variable("g", shape=[n_out])
            b = tf.get_variable("b", shape=[n_out])

            x = tf.matmul(x, v)
            scaling = g / tf.sqrt(tf.reduce_sum(tf.square(v), axis=0))

            scaling = tf.reshape(scaling, shape=[1, n_out])
            b = tf.reshape(b, shape=[1, n_out])

            return tf.multiply(scaling, x) + b


def logsumexp(x):
    """
    Numerically stable log_sum_exp implementation that prevents overflow.
    Taken from https://github.com/openai/pixel-cnn
    """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)

    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))


def logsoftmax(x):
    """
    Numerically stable log_softmax implementation that prevents overflow.
    Taken from https://github.com/openai/pixel-cnn
    """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keep_dims=True)

    return x - m - tf.log(tf.reduce_sum(tf.exp(x - m), axis, keep_dims=True))


def batch_norm(x, is_training, decay=0.99, epsilon=1e-3, center=False, scope='batch_norm'):
    """
    Batch normalization layer

      This was implemented while referring to the following papers/resources:
      https://arxiv.org/pdf/1502.03167.pdf
      https://arxiv.org/pdf/1603.09025.pdf
      http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    """
    with tf.variable_scope(scope):

        is_conv = False
        if len(x.get_shape()) == 4:  # convolution layer
            xh = x.get_shape()[1].value
            xw = x.get_shape()[2].value
            xch = x.get_shape()[3].value

            x = tf.transpose(x, perm=[0,3,1,2])
            x = tf.contrib.layers.flatten(x)
            is_conv = True

        # number of features in input matrix
        dim = x.get_shape()[1].value
        # scaling coefficients
        gamma = tf.get_variable('gamma', [dim], initializer=tf.constant_initializer(1.0), trainable=True)
        # offset coefficients (if required)
        if center:
            beta = tf.get_variable('beta', [dim], initializer=tf.constant_initializer(0), trainable=True)
        else:
            beta = None

        # population mean variable (for prediction)
        popmean = tf.get_variable('pop_mean', shape=[dim], initializer=tf.constant_initializer(0.0),
            trainable=False)
        # population variance variable (for prediction)
        popvar = tf.get_variable('pop_var', shape=[dim], initializer=tf.constant_initializer(1.0),
            trainable=False)

        # compute batch mean and variance
        batch_mean, batch_var = tf.nn.moments(x, axes=[0])

        def update_and_train():
            # update popmean and popvar using moving average of batch_mean and batch_var
            pop_mean_new = popmean * decay + batch_mean * (1 - decay)
            pop_var_new = popvar * decay + batch_var * (1 - decay)
            with tf.control_dependencies([popmean.assign(pop_mean_new), popvar.assign(pop_var_new)]):
                # batch normalization
                bn = tf.nn.batch_normalization(x, mean=batch_mean, variance=batch_var,
                    offset=beta, scale=gamma, variance_epsilon=epsilon)

                if is_conv:
                    bn = tf.reshape(bn, shape=[-1,xch,xh,xw])
                    bn = tf.transpose(bn, perm=[0,2,3,1])

                return bn

        def predict():
            # batch normalization (using population moments)
            bn = tf.nn.batch_normalization(x, mean=popmean, variance=popvar,
                offset=beta, scale=gamma, variance_epsilon=epsilon)

            if is_conv:
                bn = tf.reshape(bn, shape=[-1, xch, xh, xw])
                bn = tf.transpose(bn, perm=[0, 2, 3, 1])

            return bn

        # conditional evaluation in tf graph
        return tf.cond(is_training, update_and_train, predict)


