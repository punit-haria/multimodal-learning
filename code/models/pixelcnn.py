import tensorflow as tf
import numpy as np



def pixel_cnn(x, n_layers, ka, kb, out_ch, n_feature_maps, scope, reuse):
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

        n_ch = n_feature_maps
        nonlinearity = tf.nn.elu

        c = conv2d_masked(x, k=ka, out_ch=n_ch, mask_type='A', bias=False, scope='layer_1', reuse=reuse)

        for i in range(n_layers):
            name  = 'residual_block_' + str(i+2)
            c = masked_residual_block(c, kb, nonlinearity, scope=name, reuse=reuse)

        c = nonlinearity(c)
        c = conv2d_masked(c, k=1, out_ch=n_ch, mask_type='B', bias=False, scope='final_1x1_a', reuse=reuse)
        c = nonlinearity(c)
        c = conv2d_masked(c, k=1, out_ch=out_ch, mask_type='B', bias=False, scope='final_1x1_b', reuse=reuse)

        return c


def conditional_pixel_cnn(x, z, n_layers, ka, kb, out_ch, n_feature_maps, concat, scope, reuse):
    """
    Conditional PixelCNN

    x: input tensor
    z: latent tensor
    concat: choice of concatenating tensors or adding them
    """
    with tf.variable_scope(scope, reuse=reuse):

        n_ch = n_feature_maps
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
            c = masked_residual_block(c, kb, nonlinearity, scope=name, reuse=reuse)

        c = nonlinearity(c)
        c = conv2d_masked(c, k=1, out_ch=n_ch, mask_type='B', bias=False, scope='final_1x1_a', reuse=reuse)
        c = nonlinearity(c)
        c = conv2d_masked(c, k=1, out_ch=out_ch, mask_type='B', bias=False, scope='final_1x1_b', reuse=reuse)

        return c


def masked_residual_block(c, k, nonlinearity, scope, reuse):
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
        c1 = conv2d_masked(c1, k=1, out_ch=half_ch, mask_type='B', bias=False, scope='1x1_a', reuse=reuse)
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





