import tensorflow as tf



def convolution_mnist(x, n_ch, n_feature_maps, n_units, n_z, init, scope):
    """
    Convolution network for use with MNIST dataset.
    """
    with tf.variable_scope(scope):

        x = tf.reshape(x, shape=[-1, 28, 28, n_ch])
        nonlinearity = tf.nn.elu

        #x = conv(x, k=3, out_ch=n_feature_maps, stride=True, scope='conv_1', reuse=reuse)
        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=True, init=init, scope='res_1')
        x = nonlinearity(x)

        x = conv_residual_block(x, k=3, n_feature_maps=n_feature_maps, nonlinearity=nonlinearity,
                                stride=False, init=init, scope='unstrided_1')
        x = nonlinearity(x)

        #x = conv(x, k=3, out_ch=n_feature_maps, stride=True, scope='conv_2', reuse=reuse)
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

        return mu, sigma


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

        #z = deconv(z, k=3, out_ch=n_feature_maps, stride=True, scope='deconv_1', reuse=reuse)
        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_1')
        z = tf.pad(z, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]])
        z = nonlinearity(z)

        #z = deconv(z, k=3, out_ch=n_ch, stride=True, scope='deconv_2', reuse=reuse)
        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_2')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_feature_maps,
                                  nonlinearity=nonlinearity, stride=False, init=init, scope='unstrided_1')
        z = nonlinearity(z)

        z = deconv_residual_block(z, k=3, n_feature_maps=n_feature_maps, out_ch=n_ch, nonlinearity=nonlinearity,
                                  stride=True, init=init, scope='res_3')
        z = tf.contrib.layers.flatten(z)

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


def conv(x, k, out_ch, stride, init, scope):
    """
    Convolution layer
    """
    with tf.variable_scope(scope):

        in_ch = x.get_shape()[3].value

        strides = [1, 2, 2, 1] if stride else [1, 1, 1, 1]
        w_shape = [k, k, in_ch, out_ch]

        if init:
            v = tf.get_variable("v", shape=w_shape, initializer=tf.random_normal_initializer(0,0.05))
            v_norm = tf.nn.l2_normalize(v.initialized_value(), dim=[0,1,2])

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

            w = tf.reshape(g, shape=[1,1,1,out_ch]) * tf.nn.l2_normalize(v, dim=[0,1,3])
            b = tf.reshape(b, shape=[1,1,1,out_ch])

            d = tf.nn.conv2d_transpose(x, w, output_shape=out_shape, strides=strides, padding='SAME')
            out_shape[0] = None
            d.set_shape(out_shape)

            return d + b


def pool(x, scope, reuse):
    """
    Max pooling layer (reduces size by 2)
    """
    with tf.variable_scope(scope, reuse=reuse):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def linear(x, n_out, init, scope):
    """
    Linear tranform
    """
    with tf.variable_scope(scope):

        n_x = x.get_shape()[-1].value

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
