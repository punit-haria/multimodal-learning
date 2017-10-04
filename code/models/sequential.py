import tensorflow as tf
import numpy as np



class GRUCell(tf.contrib.rnn.RNNCell):
    """
    Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Taken from https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/rnn_cell_impl.py
    """
    def __init__(self, num_units, activation, reuse=None, kernel_initializer=None, bias_initializer=None):
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


    def call(self, inputs, state):

        with tf.variable_scope("gates"):

            # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.

            # note: state = h_{t-1}, inputs = x_t

            bias_ones = self._bias_initializer

            if self._bias_initializer is None:
                dtype = [a.dtype for a in [inputs, state]][0]
                bias_ones = tf.constant_initializer(1.0, dtype=dtype)

            value = tf.sigmoid(self._gru_linear([inputs, state], 2 * self._num_units, init, scope))

            #value = tf.sigmoid(_linear([inputs, state], 2 * self._num_units,
            #                                 True, bias_ones, self._kernel_initializer))

            r, u = tf.split(value=value, num_or_size_splits=2, axis=1)   # r_t, z_t

        with tf.variable_scope("candidate"):
            c = self._activation(_linear([inputs, r * state], self._num_units, True,
                                         self._bias_initializer, self._kernel_initializer))

        new_h = u * state + (1 - u) * c

        return new_h, new_h


    def _gru_linear(self, args, n_out, init, scope):

        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            total_arg_size += shape[1].value

        with tf.variable_scope(scope):

            n_x = total_arg_size
            x = tf.concat(args, axis=1)

            if init:
                v = tf.get_variable("v", shape=[n_x, n_out], initializer=tf.random_normal_initializer(0, 0.05))
                v_norm = tf.nn.l2_normalize(v.initialized_value(), dim=0)

                t = tf.matmul(x, v_norm)
                mu_t, var_t = tf.nn.moments(t, axes=0)

                inv = 1 / tf.sqrt(var_t + 1e-10)
                _ = tf.get_variable("g", initializer=inv)
                _ = tf.get_variable("b", initializer=-mu_t * inv) # maybe initialize with constant(1.0) for z_t, r_t ??

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



def linear_test(x, n_out, init, scope):
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




def _linear(args, output_size, bias, bias_initializer=None, kernel_initializer=None):
    """
    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
    Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    """

    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.

    scope = tf.get_variable_scope()

    with tf.variable_scope(scope) as outer_scope:

        weights = tf.get_variable(_WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype,
                                  initializer=kernel_initializer)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:

            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
            biases = tf.get_variable(_BIAS_VARIABLE_NAME, [output_size], dtype=dtype,
                                     initializer=bias_initializer)

        return nn_ops.bias_add(res, biases)

