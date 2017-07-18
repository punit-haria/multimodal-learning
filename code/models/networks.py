import tensorflow as tf
import numpy as np



def pixel_cnn(x, n_layers, k, out_ch, scope, reuse):
    """
    PixelCNN network based on architectures specified in:
        https://arxiv.org/abs/1601.06759
        https://arxiv.org/abs/1701.05517

    x: input tensor
    n_layers: number of layers in the network
    k: width of convolution filter (note: height = width)
    out_ch: number of output channels
    """
    with tf.variable_scope(scope, reuse=reuse):

        if n_layers == 1:
            n_ch = out_ch
        else:
            n_ch = 16

        nonlinearity = tf.nn.elu  # USE GATED ACTIVATION UNIT INSTEAD?  (SEE CONDITIONAL PIXELCNN PAPER)

        c = conv2d_masked(x, k, n_ch, mask_type='A', bias=False, scope='layer_1', reuse=reuse)

        for i in range(n_layers-1):

            if i == n_layers - 2:
                n_ch = out_ch

            name  = 'layer_' + str(i+2)
            c = nonlinearity(c)
            c = conv2d_masked(c, k, n_ch, mask_type='B', bias=False, scope=name, reuse=reuse)

        return nonlinearity(c)


def conditional_pixel_cnn(x, z, n_layers, out_ch, scope, reuse):
    """
    Conditional PixelCNN

    x: input tensor
    z: latent tensor
    """
    with tf.variable_scope(scope, reuse=reuse):

        if n_layers == 1:
            n_ch = out_ch
        else:
            n_ch = 16

        k = 3
        nonlinearity = tf.nn.elu

        c = tf.concat([x, z], axis=3)
        c = conv2d_masked(c, k, n_ch, mask_type='A', bias=False, scope='layer_1x', reuse=reuse)

        for i in range(n_layers-1):

            name = 'layer_' + str(i + 2)

            c = nonlinearity(c)

            if i == n_layers - 2:  # last layer
                c = conv2d_masked(c, k=k, out_ch=out_ch, mask_type='B', bias=False, scope=name, reuse=reuse)

            else:
                inter_ch = n_ch // 2
                c1 = conv2d_masked(c, k=1, out_ch=inter_ch, mask_type='B', bias=False, scope=name+'_1x1_a', reuse=reuse)
                c1 = nonlinearity(c1)
                c1 = conv2d_masked(c1, k=k, out_ch=inter_ch, mask_type='B', bias=False, scope=name, reuse=reuse)
                c1 = nonlinearity(c1)
                c1 = conv2d_masked(c1, k=1, out_ch=n_ch, mask_type='B', bias=False, scope=name+'_1x1_b', reuse=reuse)
                c = c1 + c

        return nonlinearity(c)


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
                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

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


def conv_pool(x, out_ch, n_convs, nonlinearity, scope, reuse):
    """
    Combined convolution and pooling layer.
    """
    with tf.variable_scope(scope, reuse=reuse):
        c = conv2d(x, out_ch, bias=True, scope="conv_1", reuse=reuse)

        for i in range(n_convs-1):
            c = nonlinearity(c)
            name = "conv_"+str(i+2)
            c = conv2d(c, out_ch, bias=False, scope=name, reuse=reuse)

        return pool(c, scope="pool", reuse=reuse)


def conv2d(x, out_ch, bias, scope, reuse):
    """
    Convolution layer
    """
    with tf.variable_scope(scope, reuse=reuse):

        k = 3
        in_ch = x.get_shape()[3].value
        w_shape = [k, k, in_ch, out_ch]
        w = tf.get_variable("w", shape=w_shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

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


def deconv_layer(x, out_ch, n_convs, nonlinearity, scope, reuse):
    """
    Multiple deconvolution layers.
    """
    with tf.variable_scope(scope, reuse=reuse):
        in_ch = x.get_shape()[3].value

        c = deconv(x, in_ch, out_ch, stride=True, bias=True, scope="conv_1", reuse=reuse)

        for i in range(n_convs-1):

            name = "conv_"+str(i+2)
            c = deconv(c, out_ch, out_ch, stride=False, bias=False, scope=name, reuse=reuse)
            c = nonlinearity(c)

        return c


def deconv(x, in_ch, out_ch, stride, bias, scope, reuse):
    """
    Deconvolution layer (transpose of convolution)
    """
    with tf.variable_scope(scope, reuse=reuse):
        batch_size = tf.shape(x)[0]
        height = x.get_shape()[1].value
        width = x.get_shape()[2].value

        k = 3
        w_shape = [k, k, out_ch, in_ch]
        w = tf.get_variable("w", shape=w_shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

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


def linear(x, n_x, n_w, scope, reuse):
    """
    Linear tranform
    """
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable("W", shape=[n_x, n_w],
                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable("b", shape=[n_w], initializer=tf.constant_initializer(0.1))

        return tf.matmul(x, w) + b



def batch_norm(x, scope, decay, epsilon, is_training, center=True, reuse=False):
    """
    Batch normalization layer

      This was implemented while referring to the following papers/resources:
      https://arxiv.org/pdf/1502.03167.pdf
      https://arxiv.org/pdf/1603.09025.pdf
      http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    """
    with tf.variable_scope(scope, reuse=reuse):
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
                return tf.nn.batch_normalization(x, mean=batch_mean, variance=batch_var,
                    offset=beta, scale=gamma, variance_epsilon=epsilon)

        def predict():
            # batch normalization (using population moments)
            return tf.nn.batch_normalization(x, mean=popmean, variance=popvar,
                offset=beta, scale=gamma, variance_epsilon=epsilon)

        # conditional evaluation in tf graph
        return tf.cond(is_training, update_and_train, predict)


