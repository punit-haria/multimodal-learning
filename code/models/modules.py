"""
Collection of commonly used modules for building larger architectures. 
"""
import tensorflow as tf


def affine_map(input, in_dim, out_dim, scope, reuse):
    """
    Affine transform.

    input: input tensor
    in_dim/out_dim: input and output dimensions
    scope: variable scope as string
    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable("W", shape=[in_dim,out_dim], 
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable("b", shape=[out_dim], initializer=tf.constant_initializer(0.1))

        if reuse == False:
            tf.summary.histogram('weights', W)
            tf.summary.histogram('biases', b)

        return tf.matmul(input,W) + b


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


