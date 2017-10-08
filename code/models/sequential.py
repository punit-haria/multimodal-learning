import tensorflow as tf



class GRUCell(tf.contrib.rnn.RNNCell):
    """
    Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Taken from https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/rnn_cell_impl.py
    and modified.
    """
    def __init__(self, num_units, activation, init, input, reuse=None):
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation
        self._init = init

        '''
        with tf.variable_scope('rnn'):
            with tf.variable_scope('multi_rnn_cell'):
                with tf.variable_scope('cell_0'):
                    with tf.variable_scope('gru_cell'):
                        with tf.variable_scope('gates'):
                            with tf.variable_scope('rt_zt'):
                                pass

                        with tf.variable_scope('candidate'):
                            with tf.variable_scope('ct'):
                                pass
        '''
        if self._init:
            with tf.variable_scope('rnn/multi_rnn_cell/cell_0/gru_cell/gates/rt_zt'):
                self._initialize_variables(input, 2 * self._num_units)

            with tf.variable_scope('rnn/multi_rnn_cell/cell_0/gru_cell/candidate/ct'):
                self._initialize_variables(input, self._num_units)


    def _initialize_variables(self, inputs, n_out):

        # inputs: batch x time x depth
        inp = tf.slice(inputs, begin=[0,0,0], size=[-1,1,-1])
        inp = tf.squeeze(inp)   # batch x depth
        state = tf.zeros(shape=[tf.shape(inp)[0], self._num_units])
        inp = tf.concat([inp, state], axis=1)

        n_x = inp.get_shape()[1].value

        v = tf.get_variable("v", shape=[n_x, n_out], initializer=tf.random_normal_initializer(0, 0.05))
        v_norm = tf.nn.l2_normalize(v.initialized_value(), dim=0)

        t = tf.matmul(inp, v_norm)
        mu_t, var_t = tf.nn.moments(t, axes=0)

        inv = 1 / tf.sqrt(var_t + 1e-10)

        _ = tf.get_variable("g", initializer=inv)
        _ = tf.get_variable("b", initializer=-mu_t * inv)


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

            r, z = tf.split(value=value, num_or_size_splits=2, axis=1)

        with tf.variable_scope("candidate"):
            c = self._activation(self.gru_linear([inputs, r * state], self._num_units, self._init, scope='ct'))

        new_h = z * state + (1 - z) * c

        return new_h, new_h


    def gru_linear(self, args, n_out, init, scope):

        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            total_arg_size += shape[1].value

        with tf.variable_scope(scope) as scope:

            scope.reuse_variables()

            print("-------------------------------", tf.get_variable_scope().name, flush=True)
            print("Init:", init, flush=True)

            n_x = total_arg_size
            x = tf.concat(args, axis=1)

            if init:
                '''
                v = tf.get_variable("v", shape=[n_x, n_out], initializer=tf.random_normal_initializer(0, 0.05))
                v_norm = tf.nn.l2_normalize(v.initialized_value(), dim=0)

                t = tf.matmul(x, v_norm)
                mu_t, var_t = tf.nn.moments(t, axes=0)

                inv = 1 / tf.sqrt(var_t + 1e-10)

                _ = tf.get_variable("g", initializer=inv)
                _ = tf.get_variable("b", initializer=-mu_t * inv)

                inv = tf.reshape(inv, shape=[1, n_out])
                mu_t = tf.reshape(mu_t, shape=[1, n_out])

                return tf.multiply(t - mu_t, inv)
                '''
                v = tf.get_variable("v", shape=[n_x, n_out])
                v_norm = tf.nn.l2_normalize(v.initialized_value(), dim=0)

                t = tf.matmul(x, v_norm)
                mu_t, var_t = tf.nn.moments(t, axes=0)

                inv = 1 / tf.sqrt(var_t + 1e-10)

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


