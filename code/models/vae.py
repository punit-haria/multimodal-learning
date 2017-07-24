import tensorflow as tf
import numpy as np
from copy import deepcopy

from models import base, cnn
from models import layers as nw


class VAE(base.Model):
    """
    Variational Auto-Encoder with fully connected encoder and decoder.
    """
    def __init__(self, arguments, name, tracker, session=None, log_dir=None, model_dir=None):
        # dictionary of model/inference arguments
        self.args = deepcopy(arguments)

        # object to track model performance (can be None)
        self.tracker = tracker
        if self.tracker is not None:
            self.tracker.create_run(run_name=name, model_name=self.__class__.__name__, parameters=self.args)

        # training steps counter
        self.n_steps = 0

        # base class constructor (initializes model)
        super(VAE, self).__init__(name=name, session=session, log_dir=log_dir, model_dir=model_dir)


    def _initialize(self,):
        # input/latent dimensions
        self.n_z = self.args['n_z']
        self.n_ch = self.args['n_channels']
        self.h = self.args['height']
        self.w = self.args['width']
        self.n_x = self.h * self.w * self.n_ch

        # input placeholders
        self.x = tf.placeholder(tf.float32, [None, self.n_x], name='x')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # encoder
        self.z_mu, self.z_sigma = self._encoder(self.x, scope='x_enc', reuse=False)

        # samples
        self.z = self._sample(self.z_mu, self.z_sigma, scope='sampler', reuse=False)

        # decoders
        self.rx, self.rx_probs = self._decoder(self.z, self.x, scope='x_dec', reuse=False)

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


    def _encoder(self, x, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            n_units = self.args['n_units']

            h1 = nw.linear(x, n_units, "layer_1", reuse=reuse)
            h1 = tf.nn.elu(h1)

            h2 = nw.linear(h1, n_units, "layer_2", reuse=reuse)
            h2 = tf.nn.elu(h2)

            mean = nw.linear(h2, self.n_z, "mean_layer", reuse=reuse)

            a3 = nw.linear(h2, self.n_z, "var_layer", reuse=reuse)
            sigma = tf.nn.softplus(a3)

            return mean, sigma


    def _decoder(self, z, x, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            n_units = self.args['n_units']

            z = nw.linear(z, n_units, "layer_1", reuse=reuse)
            z = tf.nn.elu(z)

            z = nw.linear(z, n_units, "layer_2", reuse=reuse)
            z = tf.nn.elu(z)

            logits = nw.linear(z, self.n_x, "logits_layer", reuse=reuse)
            probs = tf.nn.sigmoid(logits)

            return logits, probs


    def _reconstruction(self, logits, labels, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):

            l1 = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), axis=1)
            return tf.reduce_mean(l1, axis=0)


    def _penalty(self, mean, std, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):

            l2 = 0.5 * tf.reduce_sum(1 + 2*tf.log(std) - tf.square(mean) - tf.square(std), axis=1)

            return tf.reduce_mean(l2, axis=0)


    def _variational_bound(self, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            return self.l1 + self.l2


    def _loss(self, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            return -(self.l1 + self.l2)


    def _optimizer(self, loss, scope='optimizer', reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            lr = self.args['learning_rate']
            step = tf.train.RMSPropOptimizer(lr).minimize(loss)

            return step


    def _sample(self, z_mu, z_sigma, scope='sampling', reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            n_samples = tf.shape(z_mu)[0]

            eps = tf.random_normal((n_samples, self.n_z))
            z = z_mu + tf.multiply(z_sigma, eps)

            return z


    def _summaries(self,):

        with tf.variable_scope("summaries", reuse=False):
            tf.summary.scalar('lower_bound', self.bound)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('reconstruction', self.l1)
            tf.summary.scalar('penalty', self.l2)

            z = tf.abs(self.z_mu)
            z = tf.reduce_max(z, axis=0)
            tf.summary.histogram('latent_activation', z)

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



class VAE_AR(VAE):

    def __init__(self, arguments, name, tracker, session=None, log_dir=None, model_dir=None):

        super(VAE_AR, self).__init__(arguments=arguments, name=name, tracker=tracker, session=session,
                                              log_dir=log_dir, model_dir=model_dir)


    def _decoder(self, z, x, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            n_units = self.args['n_units']
            n_layers = self.args['n_pixelcnn_layers']
            n_fmaps = self.args['n_feature_maps']
            concat = self.args['concat']

            z = nw.linear(z, n_units, "layer_1", reuse=reuse)
            z = tf.nn.elu(z)

            z = nw.linear(z, self.n_x, "layer_2", reuse=reuse)
            z = tf.nn.elu(z)

            z = tf.reshape(z, shape=[-1, self.h, self.w, self.n_ch])
            x = tf.reshape(x, shape=[-1, self.h, self.w, self.n_ch])

            rx = nw.conditional_pixel_cnn(x, z, n_layers, ka=7, kb=3, out_ch=self.n_ch, n_feature_maps=n_fmaps,
                                              concat=concat, scope='pixel_cnn', reuse=reuse)

            logits = tf.reshape(rx, shape=[-1, self.n_x])

            probs = tf.nn.sigmoid(logits)

            return logits, probs


    def reconstruct(self, x):
        """
        Synthesize images autoregressively.
        """
        n_pixels = self.args['n_conditional_pixels']

        z = self.encode(x, mean=False)
        x = self._autoregressive_sampling(z, x, n_pixels)

        return x


    def decode(self, z):
        """
        Decodes z.
        """
        x = np.random.rand(z.shape[0], self.n_x)
        x = self._autoregressive_sampling(z, x, n_pixels=0)

        return x


    def decode_mode(self, x, z=None):

        if z is None:
            feed = {self.x: x, self.is_training: False}
            z = self.sess.run(self.z, feed_dict=feed)

        feed = {self.z: z, self.x: x, self.is_training: False}
        probs = self.sess.run(self.rx_probs, feed_dict=feed)

        return np.rint(probs)


    def _autoregressive_sampling(self, z, x, n_pixels):
        """
        Synthesize images autoregressively.
        """
        def _locate_2d(idx, w):
            pos = idx + 1
            r = np.ceil(pos / w)
            c = pos - (r-1)*w

            return int(r-1), int(c-1)

        h = self.h
        w = self.w
        ch = self.n_ch
        n_x = h * w * ch

        remain = h*w - n_pixels

        x = x.copy()
        for i in range(remain):
            feed = {self.z: z, self.x: x, self.is_training: False}
            probs = self.sess.run(self.rx_probs, feed_dict=feed)
            probs = np.reshape(probs, newshape=[-1, h, w, ch])

            hp, wp = _locate_2d(n_pixels + i, w)

            x = np.reshape(x, newshape=[-1, h, w, ch])
            x[:, hp, wp, :] = np.random.binomial(n=1, p=probs[:, hp, wp, :])
            x = np.reshape(x, newshape=[-1, n_x])

        return x




class VAE_CNN(VAE):

    def __init__(self, arguments, name, tracker, session=None, log_dir=None, model_dir=None):

        super(VAE_CNN, self).__init__(arguments=arguments, name=name, tracker=tracker, session=session,
                                              log_dir=log_dir, model_dir=model_dir)


    def _encoder(self, x, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            n_units = self.args['n_units']
            n_fmaps = self.args['n_feature_maps']

            mu, sigma = cnn.convolution_mnist(x, n_ch=self.n_ch, n_feature_maps=n_fmaps, n_units=n_units,
                                             n_z=self.n_z, scope='conv_network', reuse=reuse)

            return mu, sigma


    def _decoder(self, z, x, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            n_units = self.args['n_units']
            n_fmaps = self.args['n_feature_maps']

            logits = cnn.deconvolution_mnist(z, n_ch=self.n_ch, n_feature_maps=n_fmaps, n_units=n_units,
                                       scope='deconv_network', reuse=reuse)

            probs = tf.nn.sigmoid(logits)

            return logits, probs


    def _loss(self, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            alpha = self.args['anneal']
            l2 = nw.freebits_penalty(self.z_mu, self.z_sigma, alpha)

            return -(self.l1 + l2)




class VAE_CNN_AR(VAE_AR):

    def __init__(self, arguments, name, tracker, session=None, log_dir=None, model_dir=None):

        super(VAE_CNN_AR, self).__init__(arguments=arguments, name=name, tracker=tracker, session=session,
                                              log_dir=log_dir, model_dir=model_dir)


    def _encoder(self, x, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            n_units = self.args['n_units']
            n_fmaps = self.args['n_feature_maps']

            mu, sigma = nw.convolution_mnist(x, n_ch=self.n_ch, n_feature_maps=n_fmaps, n_units=n_units, n_z=self.n_z,
                                 scope='conv_network', reuse=reuse)

            return mu, sigma


    def _decoder(self, z, x, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            n_units = self.args['n_units']
            n_fmaps = self.args['n_feature_maps']
            n_layers = self.args['n_pixelcnn_layers']

            z = nw.deconvolution_mnist(z, n_ch=self.n_ch, n_feature_maps=n_fmaps, n_units=n_units,
                                       scope='deconv_network', reuse=reuse)

            x = tf.reshape(x, shape=[-1, self.h, self.w, self.n_ch])

            if self.args['conditional']:
                concat = self.args['concat']
                rx = nw.conditional_pixel_cnn(x, z, n_layers, ka=7, kb=3, out_ch=self.n_ch, n_feature_maps=n_fmaps,
                                              concat=concat, scope='pixel_cnn', reuse=reuse)
            else:
                rx = nw.pixel_cnn(z, n_layers, ka=7, kb=3, out_ch=self.n_ch, n_feature_maps=n_fmaps,
                                  scope='pixel_cnn', reuse=reuse)

            logits = tf.reshape(rx, shape=[-1, self.n_x])
            probs = tf.nn.sigmoid(logits)

            return logits, probs


    def _loss(self, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            alpha = self.args['anneal']
            l2 = nw.freebits_penalty(self.z_mu, self.z_sigma, alpha)

            return -(self.l1 + l2)






