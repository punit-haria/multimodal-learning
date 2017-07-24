import tensorflow as tf
import numpy as np
from copy import deepcopy

from models import base, cnn
from models import layers as nw
from models import flows as nf


class VAE(base.Model):
    """
    Variational Auto-Encoder with fully connected encoder and decoder.
    """
    def __init__(self, arguments, name, tracker, init_minibatch, session=None, log_dir=None, model_dir=None):
        # dictionary of model/inference arguments
        self.args = deepcopy(arguments)

        # sample minibatch for weight initialization
        self.init_minibatch = init_minibatch

        # object to track model performance (can be None)
        self.tracker = tracker
        if self.tracker is not None:
            self.tracker.create_run(run_name=name, model_name=self.__class__.__name__, parameters=self.args)

        # training steps counter
        self.n_steps = 0

        # base class constructor (initializes model)
        super(VAE, self).__init__(name=name, session=session, log_dir=log_dir, model_dir=model_dir)


    def _initialize(self):

        # input and latent dimensions
        self.n_z = self.args['n_z']
        self.n_ch = self.args['n_channels']
        self.h = self.args['height']
        self.w = self.args['width']
        self.n_x = self.h * self.w * self.n_ch

        # placeholders
        self.x = tf.placeholder(tf.float32, [None, self.n_x], name='x')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # data-dependent weight initialization (Salisman, Kingma - 2016)
        x_init = tf.constant(self.init_minibatch, tf.float32)
        self._model(x_init, init=True)

        # variational autoencoder
        self.z_mu, self.z_sigma, self.z, self.rx, self.rx_probs = self._model(self.x, init=False)

        # reconstruction and penalty terms
        self.l1 = self._reconstruction(logits=self.rx, labels=self.x, scope='reconstruction')
        self.l2 = self._penalty(mean=self.z_mu, std=self.z_sigma, scope='penalty')

        # training and test bounds
        self.bound = self._variational_bound(scope='lower_bound')

        # loss function
        self.loss = self._loss(scope='loss')

        # optimizer
        self.step = self._optimizer(self.loss)

        # summary variables
        self.summary = self._summaries()


    def _model(self, x, init):

        with tf.variable_scope('autoencoder') as scope:

            if not init:
                scope.reuse_variables()

            z_mu, z_sigma = self._encoder(x, init=init, scope='x_enc')
            z = self._sample(z_mu, z_sigma, scope='sampler')
            rx, rx_probs = self._decoder(z, x, init=init, scope='x_dec')

            return z_mu, z_sigma, z, rx, rx_probs


    def _encoder(self, x, init, scope):

        with tf.variable_scope(scope):
            n_units = self.args['n_units']

            h1 = cnn.linear(x, n_units, init=init, scope="layer_1")
            h1 = tf.nn.elu(h1)

            h2 = cnn.linear(h1, n_units, init=init, scope="layer_2")
            h2 = tf.nn.elu(h2)

            mean = cnn.linear(h2, self.n_z, init=init, scope="mean_layer")

            a3 = cnn.linear(h2, self.n_z, init=init, scope="var_layer")
            sigma = tf.nn.softplus(a3)

            return mean, sigma


    def _sample(self, z_mu, z_sigma, scope='sampling'):

        with tf.variable_scope(scope):
            n_samples = tf.shape(z_mu)[0]

            eps = tf.random_normal((n_samples, self.n_z))
            z = z_mu + tf.multiply(z_sigma, eps)

            return z


    def _decoder(self, z, x, init, scope):

        with tf.variable_scope(scope):
            n_units = self.args['n_units']

            z = cnn.linear(z, n_units, init=init, scope="layer_1")
            z = tf.nn.elu(z)

            z = cnn.linear(z, n_units, init=init, scope="layer_2")
            z = tf.nn.elu(z)

            logits = cnn.linear(z, self.n_x, init=init, scope="logits_layer")
            probs = tf.nn.sigmoid(logits)

            return logits, probs


    def _reconstruction(self, logits, labels, scope):

        with tf.variable_scope(scope):

            l1 = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), axis=1)
            return tf.reduce_mean(l1, axis=0)


    def _penalty(self, mean, std, scope):

        with tf.variable_scope(scope):

            l2 = 0.5 * tf.reduce_sum(1 + 2*tf.log(std) - tf.square(mean) - tf.square(std), axis=1)

            return tf.reduce_mean(l2, axis=0)


    def _variational_bound(self, scope):

        with tf.variable_scope(scope):
            return self.l1 + self.l2


    def _loss(self, scope):

        with tf.variable_scope(scope):
            return -(self.l1 + self.l2)


    def _optimizer(self, loss, scope='optimizer'):

        with tf.variable_scope(scope):
            lr = self.args['learning_rate']
            step = tf.train.RMSPropOptimizer(lr).minimize(loss)

            return step


    def _summaries(self,):

        with tf.variable_scope("summaries"):
            tf.summary.scalar('lower_bound', self.bound)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('reconstruction', self.l1)
            tf.summary.scalar('penalty', self.l2)

            return tf.summary.merge_all()


    def _track(self, terms, prefix):

        if self.tracker is not None:

            for name, term in terms.items():
                self.tracker.add(i=self.n_steps, value=term, series_name=prefix+name, run_name=self.name)


    def train(self, x):
        """
        Performs single training step.
        """
        feed = {self.x: x, self.is_training: True}
        outputs = [self.summary, self.step, self.bound, self.loss, self.l1, self.l2]

        summary, _, bound, loss, reconstruction, penalty = self.sess.run(outputs, feed_dict=feed)

        # track performance
        terms = {'lower_bound': bound, 'loss': loss, 'reconstruction': reconstruction, 'penalty': penalty}
        self._track(terms, prefix='train_')
        self.tr_writer.add_summary(summary, self.n_steps)

        self.n_steps = self.n_steps + 1


    def test(self, x):
        """
        Computes lower bound on test data.
        """
        feed = {self.x: x, self.is_training: False}
        outputs = [self.summary, self.bound, self.loss, self.l1, self.l2]

        summary, bound, loss, reconstruction, penalty  = self.sess.run(outputs, feed_dict=feed)

        # track performance
        terms = {'lower_bound': bound, 'loss': loss, 'reconstruction': reconstruction, 'penalty': penalty}
        self._track(terms, prefix='test_')
        self.te_writer.add_summary(summary, self.n_steps)


    def reconstruct(self, x):
        """
        Reconstruct x.
        """
        feed = {self.x: x, self.is_training: False}
        return self.sess.run(self.rx_probs, feed_dict=feed)


    def encode(self, x, mean=False):
        """
        Encode x.
        """
        feed = {self.x: x, self.is_training: False}
        if mean:
            return self.sess.run(self.z_mu, feed_dict=feed)
        else:
            return self.sess.run(self.z, feed_dict=feed)


    def decode(self, z):
        """
        Decodes z.
        """
        feed = {self.z: z, self.is_training: False}
        return self.sess.run(self.rx_probs, feed_dict=feed)


    def sample_prior(self, n_samples):
        """
        Samples z from prior distribution.
        """
        return np.random.normal(size=[n_samples, self.n_z])



class VAE_CNN(VAE):

    def __init__(self, arguments, name, tracker, init_minibatch, session=None, log_dir=None, model_dir=None):

        super(VAE_CNN, self).__init__(arguments=arguments, name=name, tracker=tracker, init_minibatch=init_minibatch,
                                      session=session, log_dir=log_dir, model_dir=model_dir)


    def _encoder(self, x, init, scope):

        with tf.variable_scope(scope):
            n_units = self.args['n_units']
            n_fmaps = self.args['n_feature_maps']

            mu, sigma = cnn.convolution_mnist(x, n_ch=self.n_ch, n_feature_maps=n_fmaps, n_units=n_units,
                                             n_z=self.n_z, init=init, scope='conv_network')

            return mu, sigma


    def _decoder(self, z, x, init, scope):

        with tf.variable_scope(scope):
            n_units = self.args['n_units']
            n_fmaps = self.args['n_feature_maps']

            logits = cnn.deconvolution_mnist(z, n_ch=self.n_ch, n_feature_maps=n_fmaps, n_units=n_units,
                                       init=init, scope='deconv_network')

            probs = tf.nn.sigmoid(logits)

            return logits, probs


    def _loss(self, scope):

        with tf.variable_scope(scope):
            alpha = self.args['anneal']
            l2 = nw.freebits_penalty(self.z_mu, self.z_sigma, alpha)

            return -(self.l1 + l2)




class VAE_NF(VAE):

    def __init__(self, arguments, name, tracker, init_minibatch, session=None, log_dir=None, model_dir=None):

        super(VAE_NF, self).__init__(arguments=arguments, name=name, tracker=tracker, init_minibatch=init_minibatch,
                                      session=session, log_dir=log_dir, model_dir=model_dir)



