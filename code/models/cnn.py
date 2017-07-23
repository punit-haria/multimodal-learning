import tensorflow as tf



def convolution_mnist(x, n_ch, n_feature_maps, n_units, n_z, scope, reuse):
    """
    Convolution network for use with MNIST dataset.
    """
    with tf.variable_scope(scope, reuse=reuse):

        x = tf.reshape(x, shape=[-1, 28, 28, n_ch])
        nonlinearity = tf.nn.elu

        #x = conv(x, k=3, out_ch=n_feature_maps, stride=True, scope='conv_1', reuse=reuse)
        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, scope='res_1', reuse=reuse)
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=False, scope='unstrided_1', reuse=reuse)
        x = nonlinearity(x)

        #x = conv(x, k=3, out_ch=n_feature_maps, stride=True, scope='conv_2', reuse=reuse)
        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, scope='res_2', reuse=reuse)
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, scope='res_3', reuse=reuse)
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=False, scope='unstrided_2', reuse=reuse)
        x = nonlinearity(x)  #

        x = tf.contrib.layers.flatten(x)

        x = linear(x, n_out=n_units, scope='linear_layer', reuse=reuse)
        x = nonlinearity(x)

        mu = linear(x, n_z, "mu_layer", reuse=reuse)

        sigma = linear(x, n_z, "sigma_layer", reuse=reuse)
        sigma = tf.nn.softplus(sigma)

        return mu, sigma


def deconvolution_mnist(z, n_ch, n_feature_maps, n_units, scope, reuse):
    """
    Deconvolution network for use with MNIST dataset.
    """
    with tf.variable_scope(scope, reuse=reuse):

        nonlinearity = tf.nn.elu

        z = linear(z, n_out=n_units, scope='mu_sigma_layer', reuse=reuse)
        z = nonlinearity(z)

        h = w = 3
        dim = h * w * n_feature_maps
        z = linear(z, n_out=dim, scope='linear_layer', reuse=reuse)
        z = nonlinearity(z)

        z = tf.reshape(z, shape=[-1, h, w, n_feature_maps])

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=False, scope='unstrided_2', reuse=reuse)
        z = nonlinearity(z)

        #z = deconv(z, k=3, out_ch=n_feature_maps, stride=True, scope='deconv_1', reuse=reuse)
        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, scope='res_1', reuse=reuse)
        z = tf.pad(z, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]])
        z = nonlinearity(z)

        #z = deconv(z, k=3, out_ch=n_ch, stride=True, scope='deconv_2', reuse=reuse)
        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, scope='res_2', reuse=reuse)
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=False, scope='unstrided_1', reuse=reuse)
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, scope='res_3', reuse=reuse)
        z = tf.contrib.layers.flatten(z)

        return z


def conv_residual_block(c, k, n_feature_maps, nonlinearity, stride, scope, reuse):
    """
    Residual Block.
    """
    with tf.variable_scope(scope, reuse=reuse):

        id = c

        c = conv_with_bias(c, k=k, out_ch=n_feature_maps, stride=False, scope='layer_1', reuse=reuse)
        c = nonlinearity(c)
        c = conv_with_bias(c, k=k, out_ch=n_feature_maps, stride=stride, scope='layer_2', reuse=reuse)

        if stride:
            id = conv(id, k=k, out_ch=n_feature_maps, stride=True, scope='identity_downsampled', reuse=reuse)

        c = c + id

        return c


def deconv_residual_block(d, k, n_feature_maps, out_ch, nonlinearity, stride, scope, reuse):
    """
    Deconvolution residual block.
    """
    with tf.variable_scope(scope, reuse=reuse):

        id = d

        d = deconv_with_bias(d, k=k, out_ch=n_feature_maps, stride=stride, scope='layer_1', reuse=reuse)
        d = nonlinearity(d)
        d = deconv_with_bias(d, k=k, out_ch=out_ch, stride=False, scope='layer_2', reuse=reuse)

        if stride:
            id = deconv(id, k=k, out_ch=out_ch, stride=True, scope='identity_upsampled', reuse=reuse)

        d = d + id

        return d


def conv(x, k, out_ch, stride, scope, reuse):
    """
    Convolution layer
    """
    with tf.variable_scope(scope, reuse=reuse):

        in_ch = x.get_shape()[3].value

        strides = [1, 2, 2, 1] if stride else [1, 1, 1, 1]
        w = weight([k, k, in_ch, out_ch])

        return tf.nn.conv2d(x, w, strides=strides, padding='SAME')


def deconv(x, k, out_ch, stride, scope, reuse):
    """
    Deconvolution layer (transpose of convolution)
    """
    with tf.variable_scope(scope, reuse=reuse):

        batch_size = tf.shape(x)[0]
        height = x.get_shape()[1].value
        width = x.get_shape()[2].value
        in_ch = x.get_shape()[3].value

        if stride:
            out_shape = [batch_size, height*2, width*2, out_ch]
            stride = [1, 2, 2, 1]
        else:
            out_shape = [batch_size, height, width, out_ch]
            stride = [1, 1, 1, 1]

        w = weight([k, k, out_ch, in_ch])

        dcv = tf.nn.conv2d_transpose(x, w, output_shape=out_shape, strides=stride, padding='SAME')

        out_shape[0] = None
        dcv.set_shape(out_shape)

        return dcv


def conv_with_bias(x, k, out_ch, stride, scope, reuse):
    """
    Convolution layer with bias.
    """
    c = conv(x, k, out_ch, stride, scope, reuse)

    with tf.variable_scope(scope, reuse=reuse):
        b = bias(out_ch)

        return c + b


def deconv_with_bias(x, k, out_ch, stride, scope, reuse):
    """
    Deconvolution layer with bias.
    """
    dcv = deconv(x, k, out_ch, stride, scope, reuse)

    with tf.variable_scope(scope, reuse=reuse):
        b = bias(out_ch)

        return dcv + b


def pool(x, scope, reuse):
    """
    Max pooling layer (reduces size by 2)
    """
    with tf.variable_scope(scope, reuse=reuse):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def linear(x, n_out, scope, reuse):
    """
    Linear tranform
    """
    with tf.variable_scope(scope, reuse=reuse):
        n_x = x.get_shape()[-1].value
        w = weight([n_x, n_out])
        b = bias(n_out)

        return tf.matmul(x, w) + b


def weight(shape, init=tf.contrib.layers.xavier_initializer, name="w"):
    """
    Initialize weight matrix.
    """
    return tf.get_variable(name, shape=shape, initializer=init())


def bias(n_out, name="b"):
    """
    Initialize bias vector.
    """
    return tf.get_variable(name, shape=[n_out], initializer=tf.constant_initializer(0.1))
