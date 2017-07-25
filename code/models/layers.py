import tensorflow as tf
import numpy as np



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

        return mu, sigma, h


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
                                  nonlinearity=nonlinearity, stride=True, init=init, scope='res_1')
        z = tf.pad(z, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]])
        z = nonlinearity(z)

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


def normalizing_flow(mu0, sigma0, h, epsilon, K, n_units, init, scope):
    """
    Normalizing flow.
    """
    with tf.variable_scope(scope):
        # NOTE: Transforms should be stacked in reverse ordering between every other transformation!!!!!!!!!!!!!!!!!!!!

        z = mu0 + tf.multiply(sigma0, epsilon)
        log_q = -tf.reduce_sum(tf.log(sigma0) + 0.5 * tf.square(epsilon) + 0.5 * np.log(2 * np.pi), axis=1)

        for i in range(K):
            m, s = made_network(z, h=h, n_units=n_units, init=init, scope='flow_'+str(i+1))
            sigma = tf.nn.sigmoid(s)

            z = tf.multiply(sigma, z) + tf.multiply(1-sigma, m)
            log_q = log_q - tf.reduce_sum(tf.log(sigma), axis=1)

        return z, log_q


def made_network(z, h, n_units, init, scope):
    """
    Masked Network (MADE) based on https://arxiv.org/abs/1502.03509
    used as single normalizing flow transform.
    """
    with tf.variable_scope(scope):

        # NOTE: Initialize network so that output s is sufficiently positive (i.e. close to +1 or +2)

        nonlinearity = tf.nn.elu
        n_z = z.get_shape()[1].value
        m = None

        z, m = ar_linear(z, n_out=n_units, m_prev=m, init=init, scope='layer_1')
        z = nonlinearity(z)

        z, m = ar_linear(z, n_out=n_units, m_prev=m, init=init, scope='layer_2_z')
        h = ar_mult(h, n_out=n_units, init=init, scope='layer_2_h')
        z = nonlinearity(z + h)

        mu, _ = ar_linear(z, n_out=n_z, m_prev=m, init=init, scope='layer_2_z')
        gate, _ = ar_linear(z, n_out=n_z, m_prev=m, init=init, scope='layer_2_z')

        return mu, gate


def ar_linear(x, n_out, m_prev, init, scope):
    """
    Masked linear transform based on MADE network (https://arxiv.org/abs/1502.03509)
    Results in autoregressive relationship between input and output.
    """
    with tf.variable_scope(scope):
        Kin = x.get_shape()[1].value
        Kout = n_out

        w = tf.get_variable("w", shape=[Kin, Kout], initializer=tf.random_normal_initializer(0, 0.05))
        b = tf.get_variable("b", shape=[Kout], initializer=tf.constant_initializer(0.1))

        if m_prev is None:
            m_prev = np.arange(Kin) + 1
            m = np.random.randint(low=1, high=Kin, size=Kout)
        else:
            m = np.random.randint(low=np.min(m_prev), high=Kin, size=Kout)

        mask = np.zeros(shape=[Kin, Kout], dtype=np.float32)
        for kin in range(Kin):
            for kout in range(Kout):
                if m[kout] >= m_prev[kin]:
                    mask[kin, kout] = 1

        w = tf.multiply(w, tf.constant(mask))
        x = tf.matmul(x, w) + b

        return x, m


def ar_mult(x, n_out, init, scope):
    """
    Matrix multiplication with simple lower triangular mask.
    Results in autoregressive relationship between input and output.
    """
    with tf.variable_scope(scope):
        n_in = x.get_shape()[1].value

        w = tf.get_variable("w", shape=[n_in, n_out], initializer=tf.random_normal_initializer(0, 0.05))
        b = tf.get_variable("b", shape=[n_out], initializer=tf.constant_initializer(0.1))

        mask = np.ones(shape=[n_in, n_out], dtype=np.float32)
        mask = np.tril(mask)  # strictly lower triangular

        w = tf.multiply(w, tf.constant(mask))
        x = tf.matmul(x, w)

        return x


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

        return mean, sigma, h


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

        logits = linear(z, n_x, init=init, scope="logits_layer")

        return logits


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


def freebits_penalty(mu, sigma, alpha):
    """
    Freebits penalty function as specified in the Inverse Autoregressive Flow paper (Kingma et al).
    """
    l2 = 0.5 * (1 + 2 * tf.log(sigma) - tf.square(mu) - tf.square(sigma))
    l2 = tf.reduce_mean(l2, axis=0)
    l2 = tf.minimum(l2, alpha)

    return tf.reduce_sum(l2)


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


