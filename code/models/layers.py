import tensorflow as tf
import numpy as np



def pixel_cnn(x, n_layers, ka, kb, out_ch, scope, reuse):
    """
    PixelCNN network based on architectures specified in:
        https://arxiv.org/abs/1601.06759
        https://arxiv.org/abs/1701.05517

    x: input tensor
    n_layers: number of layers in the network
    ka/kb: widths of mask A and B convolution filters
    out_ch: number of output channels
    """
    with tf.variable_scope(scope, reuse=reuse):

        n_ch = 32
        nonlinearity = tf.nn.elu

        c = conv2d_masked(x, k=ka, out_ch=n_ch, mask_type='A', bias=False, scope='layer_1', reuse=reuse)
        #c = nonlinearity(c)

        for i in range(n_layers):
            name  = 'residual_block_' + str(i+2)
            c = residual_block(c, kb, nonlinearity, scope=name, reuse=reuse)

        c = nonlinearity(c)
        c = conv2d_masked(c, k=1, out_ch=n_ch, mask_type='B', bias=False, scope='final_1x1_a', reuse=reuse)
        c = nonlinearity(c)
        c = conv2d_masked(c, k=1, out_ch=out_ch, mask_type='B', bias=False, scope='final_1x1_b', reuse=reuse)

        return c


def conditional_pixel_cnn(x, z, n_layers, ka, kb, out_ch, concat, scope, reuse):
    """
    Conditional PixelCNN

    x: input tensor
    z: latent tensor
    concat: choice of concatenating tensors or adding them
    """
    with tf.variable_scope(scope, reuse=reuse):

        n_ch = 32
        nonlinearity = tf.nn.elu

        if concat:
            c = tf.concat([x, z], axis=3)
            c = conv2d_masked(c, k=ka, out_ch=n_ch, mask_type='A', bias=False, scope='layer_1', reuse=reuse)
        else:
            cx = conv2d_masked(x, k=ka, out_ch=n_ch, mask_type='A', bias=False, scope='layer_1x', reuse=reuse)
            cz = conv2d(z, k=ka, out_ch=n_ch, bias=True, scope='layer_1z', reuse=reuse)
            c = cx + cz

        for i in range(n_layers):
            name = 'residual_block_' + str(i + 2)
            c = residual_block(c, kb, nonlinearity, scope=name, reuse=reuse)

        c = nonlinearity(c)
        c = conv2d_masked(c, k=1, out_ch=n_ch, mask_type='B', bias=False, scope='final_1x1_a', reuse=reuse)
        c = nonlinearity(c)
        c = conv2d_masked(c, k=1, out_ch=out_ch, mask_type='B', bias=False, scope='final_1x1_b', reuse=reuse)

        return c


def residual_block(c, k, nonlinearity, scope, reuse):
    """
    Residual Block for PixelCNN. See https://arxiv.org/abs/1601.06759
    c: input tensor
    k: filter size
    nonlinearity: activation function
    """
    with tf.variable_scope(scope, reuse=reuse):

        n_ch = c.get_shape()[3].value
        half_ch = n_ch // 2
        c1 = nonlinearity(c)
        c1 = conv2d_masked(c, k=1, out_ch=half_ch, mask_type='B', bias=False, scope='1x1_a', reuse=reuse)
        c1 = nonlinearity(c1)
        c1 = conv2d_masked(c1, k=k, out_ch=half_ch, mask_type='B', bias=False, scope='conv', reuse=reuse)
        c1 = nonlinearity(c1)
        c1 = conv2d_masked(c1, k=1, out_ch=n_ch, mask_type='B', bias=False, scope='1x1_b', reuse=reuse)
        c = c1 + c

        return c


def conv2d_masked(x, k, out_ch, mask_type, bias, scope, reuse):
    """
    Masked 2D convolution

    x: input tensor
    k: convolution window
    out_ch: number of output channels
    mask_type: mask type 'A' or 'B' (see https://arxiv.org/abs/1601.06759)
    bias: incorporate bias? (True/False)
    """
    with tf.variable_scope(scope, reuse=reuse):

        assert k % 2 == 1  # check that k is odd
        in_ch = x.get_shape()[3].value  # number of input channels

        w = tf.get_variable("W", shape=[k, k, in_ch, out_ch],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        # create mask
        if k == 1:
            assert mask_type == 'B'
            mask = np.ones(shape=[k, k, in_ch, out_ch], dtype=np.float32)

        else:
            mask = np.zeros(shape=[k, k, in_ch, out_ch], dtype=np.float32)
            half_k = (k // 2) + 1
            for i in range(half_k):
                for j in range(k):
                    if i < half_k - 1 or j < half_k - 1:
                        mask[i,j,:,:] = 1

            # mask type
            if mask_type == 'A':
                mask[half_k-1,half_k-1,:,:] = 0
            elif mask_type == 'B':
                mask[half_k - 1, half_k - 1, :, :] = 1
            else:
                raise Exception("Masking type not implemented..")

        # incorporate bias term
        if bias:
            b = tf.get_variable("b", shape=[out_ch], initializer=tf.constant_initializer(0.1))
        else:
            b = 0

        # mask filter and apply convolution
        w_masked = tf.multiply(w, tf.constant(mask))
        c = tf.nn.conv2d(x, w_masked, strides=[1,1,1,1], padding='SAME') + b

        return c


def conv_pool(x, k, out_ch, n_convs, nonlinearity, scope, reuse):
    """
    Combined convolution and pooling layer.
    """
    with tf.variable_scope(scope, reuse=reuse):
        c = conv2d(x, k, out_ch, bias=True, scope="conv_1", reuse=reuse)
        c = nonlinearity(c)

        for i in range(n_convs-1):
            name = "conv_"+str(i+2)
            c = conv2d(c, k, out_ch, bias=False, scope=name, reuse=reuse)
            c = nonlinearity(c)

        return pool(c, scope="pool", reuse=reuse)


def conv2d(x, k, out_ch, bias, scope, reuse):
    """
    Convolution layer
    """
    with tf.variable_scope(scope, reuse=reuse):

        in_ch = x.get_shape()[3].value
        w_shape = [k, k, in_ch, out_ch]
        w = tf.get_variable("w", shape=w_shape, initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        if bias:
            b = tf.get_variable("b", shape=[out_ch], initializer=tf.constant_initializer(0.1))
        else:
            b = 0

        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME') + b


def pool(x, scope, reuse):
    """
    Max pooling layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def deconv_layer(x, k, out_ch, n_convs, nonlinearity, scope, reuse):
    """
    Multiple deconvolution layers.
    """
    with tf.variable_scope(scope, reuse=reuse):
        in_ch = x.get_shape()[3].value

        c = deconv(x, k, in_ch, out_ch, stride=True, bias=True, scope="conv_1", reuse=reuse)
        c = nonlinearity(c)

        for i in range(n_convs-1):

            name = "conv_"+str(i+2)
            c = deconv(c, k, out_ch, out_ch, stride=False, bias=False, scope=name, reuse=reuse)
            c = nonlinearity(c)

        return c


def deconv(x, k, in_ch, out_ch, stride, bias, scope, reuse):
    """
    Deconvolution layer (transpose of convolution)
    """
    with tf.variable_scope(scope, reuse=reuse):
        batch_size = tf.shape(x)[0]
        height = x.get_shape()[1].value
        width = x.get_shape()[2].value

        w_shape = [k, k, out_ch, in_ch]
        w = tf.get_variable("w", shape=w_shape, initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        if bias:
            b = tf.get_variable("b", shape=[out_ch], initializer=tf.constant_initializer(0.1))
        else:
            b = 0

        if stride:
            out_shape = [batch_size, height*2, width*2, out_ch]
            stride = [1, 2, 2, 1]
        else:
            out_shape = [batch_size, height, width, out_ch]
            stride = [1, 1, 1, 1]

        dcv = tf.nn.conv2d_transpose(x, w, output_shape=out_shape, strides=stride, padding='SAME')
        out_shape[0] = None
        dcv.set_shape(out_shape)

        return dcv + b


def linear(x, n_out, scope, reuse):
    """
    Linear tranform
    """
    with tf.variable_scope(scope, reuse=reuse):
        n_x = x.get_shape()[-1].value

        w = tf.get_variable("W", shape=[n_x, n_out],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        b = tf.get_variable("b", shape=[n_out], initializer=tf.constant_initializer(0.1))

        return tf.matmul(x, w) + b



def batch_norm(x, is_training, decay=0.99, epsilon=1e-3, center=False, scope='batch_norm', reuse=False):
    """
    Batch normalization layer

      This was implemented while referring to the following papers/resources:
      https://arxiv.org/pdf/1502.03167.pdf
      https://arxiv.org/pdf/1603.09025.pdf
      http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    """
    with tf.variable_scope(scope, reuse=reuse):

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


