import tensorflow as tf
import numpy as np
from models import base


class VAE(base.Model):
    """
    Variational Auto-Encoder with 2 inputs

    Arguments:
    n_x1, n_x2, n_z: dimensionality of input and latent variables
    learning_rate: optimizer learning_rate
    n_units: number of hidden units in fully-connected layers
    """
    def __init__(self, arguments, name="VAE", session=None, log_dir=None, model_dir=None):
        # dictionary of model/inference arguments
        self.args = arguments

        # training steps counter
        self.n_steps = 0

        # base class constructor (initializes model)
        super(VAE, self).__init__(name=name, session=session, log_dir=log_dir, model_dir=model_dir)


    def _initialize(self,):
        # input/latent dimensions
        n_x1 = self.args['n_x1']
        n_x2 = self.args['n_x2']
        n_z = self.args['n_z']

        # input placeholders
        self.x1 = tf.placeholder(tf.float32, [None, n_x1], name='x1')
        self.x2 = tf.placeholder(tf.float32, [None, n_x2], name='x2')
        self.x1p = tf.placeholder(tf.float32, [None, n_x1], name='x1_paired')
        self.x2p = tf.placeholder(tf.float32, [None, n_x2], name='x2_paired')

        # encoders
        self.z1_mu, self.z1_var = self._encoder(self.x1, n_x1, n_z, scope='x1_enc', reuse=False)
        self.z2_mu, self.z2_var = self._encoder(self.x2, n_x2, n_z, scope='x2_enc', reuse=False)
        z1p_mu, z1p_var = self._encoder(self.x1p, n_x1, n_z, scope='x1_enc', reuse=True)
        z2p_mu, z2p_var = self._encoder(self.x2p, n_x2, n_z, scope='x2_enc', reuse=True)

        # constrain paired encoder
        self.z12_mu, self.z12_var = self._constrain(z1p_mu, z1p_var, z2p_mu, z2p_var, scope='x1x2_enc', reuse=False)

        # samples
        self.z1 = self._sample(self.z1_mu, self.z1_var, n_z, scope='sample_1', reuse=False)
        self.z2 = self._sample(self.z2_mu, self.z2_var, n_z, scope='sample_2', reuse=False)
        self.z12 = self._sample(self.z12_mu, self.z12_var, n_z, scope='sample_12', reuse=False)
        self.z1p = self._sample(z1p_mu, z1p_var, n_z, scope='sample_1p', reuse=False)
        self.z2p = self._sample(z2p_mu, z2p_var, n_z, scope='sample_2p', reuse=False)

        # decoders
        self.rx1_1, self.rx1_1_probs = self._decoder(self.z1, n_z, n_x1, scope='x1_dec', reuse=False)
        self.rx1_2, self.rx1_2_probs = self._decoder(self.z2, n_z, n_x1, scope='x1_dec', reuse=True)
        self.rx2_1, self.rx2_1_probs = self._decoder(self.z1, n_z, n_x2, scope='x2_dec', reuse=False)
        self.rx2_2, self.rx2_2_probs = self._decoder(self.z2, n_z, n_x2, scope='x2_dec', reuse=True)
        self.rx1p, self.rx1p_probs = self._decoder(self.z12, n_z, n_x1, scope='x1_dec', reuse=True)
        self.rx2p, self.rx2p_probs = self._decoder(self.z12, n_z, n_x2, scope='x2_dec', reuse=True)

        # additional decoders
        self.rx1p_1, self.rx1p_1_probs = self._decoder(self.z1p, n_z, n_x1, scope='x1_dec', reuse=True)
        self.rx1p_2, self.rx1p_2_probs = self._decoder(self.z2p, n_z, n_x1, scope='x1_dec', reuse=True)
        self.rx2p_1, self.rx2p_1_probs = self._decoder(self.z1p, n_z, n_x2, scope='x2_dec', reuse=True)
        self.rx2p_2, self.rx2p_2_probs = self._decoder(self.z2p, n_z, n_x2, scope='x2_dec', reuse=True)

        # choice of variational bounds
        self.l_x1 = self._marginal_bound(self.rx1_1, self.x1, self.z1_mu, self.z1_var, scope='marginal_x1')
        self.l_x2 = self._marginal_bound(self.rx2_2, self.x2, self.z2_mu, self.z2_var, scope='marginal_x2')
        self.t_x1 = self._translation_bound(self.rx1p_2, self.x1p, scope='translate_to_x1')
        self.t_x2 = self._translation_bound(self.rx2p_1, self.x2p, scope='translate_to_x2')
        self.l_x1x2 = self._joint_bound(self.rx1p, self.x1p, self.rx2p, self.x2p, self.z12_mu, self.z12_var)

        # training and test bounds
        self.bound = self._variational_bound()

        # loss function
        self.loss = -self.bound

        # optimizer
        self.step = self._optimizer(self.loss)


    def _encoder(self, x, n_x, n_z, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            n_units = self.args['n_units']

            a1 = self._linear(x, n_x, n_units, "layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)

            a2 = self._linear(h1, n_units, n_units, "layer_2", reuse=reuse)
            h2 = tf.nn.relu(a2)

            mean = self._linear(h2, n_units, n_z, "mean_layer", reuse=reuse)

            a3 = self._linear(h2, n_units, n_z, "var_layer", reuse=reuse)
            var = tf.nn.softplus(a3)

            return mean, var


    def _decoder(self, z, n_z, n_x, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            n_units = self.args['n_units']

            a1 = self._linear(z, n_z, n_units, "layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)

            a2 = self._linear(h1, n_units, n_units, "layer_2", reuse=reuse)
            h2 = tf.nn.relu(a2)

            logits = self._linear(h2, n_units, n_x, "layer_3", reuse=reuse)
            probs = tf.nn.sigmoid(logits)

            return logits, probs


    def _variational_bound(self,):
        joint = tf.concat([self.l_x1x2, self.l_x1, self.l_x2], axis=0)

        return tf.reduce_mean(joint, axis=0)


    def _marginal_bound(self, logits, labels, mean, var, scope='marginal_bound', reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            l1 = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                labels=labels), axis=1)

            l2 = 0.5 * tf.reduce_sum(1 + tf.log(var) - tf.square(mean) - var, axis=1)

            return l1 + l2


    def _joint_bound(self, x1_logits, x1_labels, x2_logits, x2_labels, mean, var, scope='joint_bound', reuse=False):

        with tf.variable_scope(scope, reuse=reuse):

            l1_x = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x1_logits, labels=x1_labels), axis=1)
            l1_y = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x2_logits, labels=x2_labels), axis=1)

            l2 = 0.5 * tf.reduce_sum(1 + tf.log(var) - tf.square(mean) - var, axis=1)

            return l1_x + l1_y + l2


    def _translation_bound(self, logits, labels, scope='translation_bound', reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            return tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), axis=1)


    def _optimizer(self, loss, scope='optimizer', reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            lr = self.args['learning_rate']
            step = tf.train.RMSPropOptimizer(lr).minimize(loss)

            return step


    def _sample(self, z_mu, z_var, n_z, scope='sampling', reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            n_samples = tf.shape(z_mu)[0]

            z_std = tf.sqrt(z_var)
            eps = tf.random_normal((n_samples, n_z))
            z = z_mu + tf.multiply(z_std, eps)

            return z


    def _constrain(self, x1_mu, x1_var, x2_mu, x2_var, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            # Computes mean and variance of the product of two Gaussians.
            x1v_inv = tf.reciprocal(x1_var)
            x2v_inv = tf.reciprocal(x2_var)
            x12_var = tf.reciprocal(x1v_inv + x2v_inv)
            xx = tf.multiply(x1v_inv, x1_mu)
            yy = tf.multiply(x2v_inv, x2_mu)
            x12_mu = tf.multiply(x12_var, xx + yy)

            return x12_mu, x12_var


    def _summaries(self,):

        with tf.variable_scope("summaries", reuse=False):
            tf.summary.scalar('variational_bound', self.bound)

            return tf.summary.merge_all()


    def _linear(self, x, n_x, n_w, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            w = tf.get_variable("W", shape=[n_x,n_w],
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            b = tf.get_variable("b", shape=[n_w], initializer=tf.constant_initializer(0.1))

            return tf.matmul(x, w) + b


    def train(self, x1, x2, x1_pairs, x2_pairs, write=True):
        """
        Performs single training step.
        """
        feed = {self.x1: x1, self.x2: x2, self.x1p: x1_pairs, self.x2p: x2_pairs}
        outputs = [self.summary, self.step, self.bound]

        summary, _, bound = self.sess.run(outputs, feed_dict=feed)
        if write:
            self.tr_writer.add_summary(summary, self.n_steps)
        self.n_steps = self.n_steps + 1

        return bound


    def test(self, x1, x2):
        """
        Computes lower bound on test data.
        """
        x1_shape = x1.shape
        x1_shape[0] = 0
        x2_shape = x2.shape
        x2_shape[0] = 0
        x1_empty = np.zeros(shape=x1_shape)
        x2_empty = np.zeros(shape=x2_shape)

        feed = {self.x1: x1_empty, self.x2: x2_empty, self.x1p: x1, self.x2p: x2}
        outputs = [self.summary, self.bound]

        summary, test_bound = self.sess.run(outputs, feed_dict=feed)
        self.te_writer.add_summary(summary, self.n_steps)

        return test_bound


    def translate_x1(self, x1):
        """
        Translate x1 to x2.
        """
        feed = {self.x1: x1}
        return self.sess.run(self.rx2_1_probs, feed_dict=feed)


    def translate_x2(self, x2):
        """
        Translate x2 to x1.
        """
        feed = {self.x2: x2}
        return self.sess.run(self.rx1_2_probs, feed_dict=feed)


    def reconstruct(self, x1, x2):
        """
        Reconstruct x1, x2 given both x1 and x2.
        """
        feed = {self.x1p: x1, self.x2p: x2}
        return self.sess.run([self.rx1p_probs, self.rx2p_probs], feed_dict=feed)


    def reconstruct_from_x1(self, x1):
        """
        Reconstruct x1, x2 given only x1.
        """
        feed = {self.x1: x1}
        return self.sess.run([self.rx1_1_probs, self.rx2_1_probs], feed_dict=feed)


    def reconstruct_from_x2(self, x2):
        """
        Reconstruct x1, x2 given only x2.
        """
        feed = {self.x2: x2}
        return self.sess.run([self.rx1_2_probs, self.rx2_2_probs], feed_dict=feed)


    def encode_x1(self, x1):
        """
        Encode x1.
        """
        feed = {self.x1: x1}
        return self.sess.run(self.z1_mu, feed_dict=feed)


    def encode_x2(self, x2):
        """
        Encode x2.
        """
        feed = {self.x2: x2}
        return self.sess.run(self.z2_mu, feed_dict=feed)


    def encode(self, x1, x2):
        """
        Encode x1 and x2 jointly.
        """
        feed = {self.x1p: x1, self.x2p: x2}
        return self.sess.run(self.z12_mu, feed_dict=feed)


    def decode(self, z):
        """
        Decodes x1 and x2.
        """
        feed = {self.z12: z}
        return self.sess.run([self.rx1p_probs, self.rx2p_probs], feed_dict=feed)



class VAETranslate(VAE):

    def __init__(self, arguments, name="VAETranslate", session=None, log_dir=None, model_dir=None):

        # base class constructor (initializes model)
        super(VAETranslate, self).__init__(arguments=arguments, name=name, session=session,
                                              log_dir=log_dir, model_dir=model_dir)

    def _variational_bound(self,):

        x1_bound = tf.reduce_mean(self.t_x1 + self.l_x2)
        x2_bound = tf.reduce_mean(self.t_x2 + self.l_x1)

        return (x1_bound + x2_bound) / 2


    def train(self, x1, x2, x1_pairs, x2_pairs, write=True):
        """
        Performs single training step.
        """
        feed = {self.x1: x1, self.x2: x2, self.x1p: x1_pairs, self.x2p: x2_pairs}
        outputs = [self.summary, self.step, self.bound]

        summary, _, bound = self.sess.run(outputs, feed_dict=feed)
        if write:
            self.tr_writer.add_summary(summary, self.n_steps)
        self.n_steps = self.n_steps + 1

        return bound


    def test(self, x1, x2, x1_pairs, x2_pairs):
        """
        Computes variational bounds on test data.
        """
        feed = {self.x1: x1, self.x2: x2, self.x1p: x1_pairs, self.x2p: x2_pairs}
        outputs = [self.summary, self.l_x1, self.l_x2, self.l_x1x2]

        summary, x1_bound, x2_bound, joint_bound = self.sess.run(outputs, feed_dict=feed)
        self.te_writer.add_summary(summary, self.n_steps)

        return x1_bound, x2_bound, joint_bound