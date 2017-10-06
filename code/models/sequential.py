import tensorflow as tf



class GRUCell(tf.contrib.rnn.RNNCell):
    """
    Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Taken from https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/rnn_cell_impl.py
    and modified.
    """
    def __init__(self, num_units, activation, init, reuse=None):
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation
        self._init = init
        self.called = False


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


    def call(self, inputs, state):

        with tf.variable_scope("gates"):

            # note: start with bias of 1.0 to not reset and not update.
            # note: state = h_{t-1}, inputs = x_t

            value = tf.sigmoid(self.gru_linear([inputs, state], 2 * self._num_units, self._init, scope='rt_zt'))

            r, z = tf.split(value=value, num_or_size_splits=2, axis=1)   # r_t, z_t

        with tf.variable_scope("candidate"):
            c = self._activation(self.gru_linear([inputs, r * state], self._num_units, self._init, scope='ct'))

        new_h = z * state + (1 - z) * c

        self.called = True

        return new_h, new_h


    def gru_linear(self, args, n_out, init, scope):

        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            total_arg_size += shape[1].value

        with tf.variable_scope(scope):

            print("-------------------------------", tf.get_variable_scope().name, flush=True)
            print("Init:", init, flush=True)

            n_x = total_arg_size
            x = tf.concat(args, axis=1)

            if init:

                v = tf.get_variable("v", shape=[n_x, n_out], initializer=tf.random_normal_initializer(0, 0.05))
                v_norm = tf.nn.l2_normalize(v.initialized_value(), dim=0)

                t = tf.matmul(x, v_norm)
                mu_t, var_t = tf.nn.moments(t, axes=0)

                inv = 1 / tf.sqrt(var_t + 1e-10)

                #_ = tf.get_variable("g", shape=[n_out], initializer=tf.random_normal_initializer(0, 0.05))
                #_ = tf.get_variable("b", shape=[n_out], initializer=tf.constant_initializer(1.0))

                _ = tf.get_variable("g", initializer=inv)
                _ = tf.get_variable("b", initializer=-mu_t * inv) # maybe initialize with constant(1.0) for z_t, r_t..

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


