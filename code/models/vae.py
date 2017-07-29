import tensorflow as tf
import numpy as np
from copy import deepcopy

from models import base
from models import layers as nw


class VAE(base.Model):
    """
    Variational Auto-Encoder
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

        # options
        self.nw_type = self.args["type"]
        self.dataset = self.args["data"]
        self.is_autoregressive = self.args["autoregressive"]
        self.is_flow = self.args["flow"]

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
        self.z_mu, self.z_sigma, self.z, log_q, self.rx, self.rx_probs = self._model(self.x, init=False)

        # reconstruction and penalty terms
        self.l1 = self._reconstruction(logits=self.rx, labels=self.x, scope='reconstruction')
        self.l2, self.log_q, self.log_p = self._penalty(mu=self.z_mu, sigma=self.z_sigma,
                                log_q=log_q, z_K=self.z, scope='penalty')

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

            z_mu, z_sigma, h = self._encoder(x, init=init, scope='x_enc')
            z, log_q = self._sample(z_mu, z_sigma, h, init=init, scope='sampler')
            rx, rx_probs = self._decoder(z, x, init=init, scope='x_dec')

            return z_mu, z_sigma, z, log_q, rx, rx_probs


    def _encoder(self, x, init, scope):

        with tf.variable_scope(scope):
            n_units = self.args['n_units']
            n_fmaps = self.args['n_feature_maps']
            extra = self.args['flow']  # extra output if using normalizing flow

            mu = sigma = h = None
            if self.nw_type == "fc":
                mu, sigma, h = nw.fc_encode(x, n_units=n_units, n_z=self.n_z, extra=extra,
                                            init=init, scope='fc_network')

            elif self.nw_type == "cnn":
                mu, sigma, h = nw.convolution_mnist(x, n_ch=self.n_ch, n_feature_maps=n_fmaps, n_units=n_units,
                                                    n_z=self.n_z, extra=extra, init=init, scope='conv_network')

            return mu, sigma, h


    def _sample(self, mu0, sigma0, h, init, scope):

        with tf.variable_scope(scope):
            n_samples = tf.shape(mu0)[0]
            epsilon = tf.random_normal((n_samples, self.n_z))

            if self.is_flow:
                n_layers = self.args['flow_layers']
                n_units = self.args['flow_units']
                flow_type = self.args['flow_type']

                z, log_q = nw.normalizing_flow(mu0, sigma0, h=h, epsilon=epsilon, K=n_layers, n_units=n_units,
                                               flow_type=flow_type, init=init, scope='normalizing_flow')

            else:
                z = mu0 + tf.multiply(sigma0, epsilon)
                log_q = None

            return z, log_q


    def _decoder(self, z, x, init, scope):

        with tf.variable_scope(scope):
            n_units = self.args['n_units']
            n_fmaps = self.args['n_feature_maps']
            n_layers = self.args['n_pixelcnn_layers']
            n_x = self.n_x
            n_ch = self.n_ch

            if self.dataset == "mnist":
                n_cats = 1
            else:
                n_cats = 256

            if self.nw_type == "fc":
                if not self.is_autoregressive:
                    n_x = n_x * n_cats

                z = nw.fc_decode(z, n_units=n_units, n_x=n_x, init=init, scope='fc_network')

            elif self.nw_type == "cnn":
                if not self.is_autoregressive:
                    n_ch = n_ch * n_cats

                z = nw.deconvolution_mnist(z, n_ch=n_ch, n_feature_maps=n_fmaps, n_units=n_units,
                                           init=init, scope='deconv_network')
            else:
                raise NotImplementedError

            if self.is_autoregressive:
                x = tf.reshape(x, shape=[-1, self.h, self.w, self.n_ch])
                z = tf.reshape(z, shape=[-1, self.h, self.w, self.n_ch])

                n_ch = self.n_ch * n_cats

                z = nw.conditional_pixel_cnn(x, z, n_layers=n_layers, out_ch=n_ch,
                                              n_feature_maps=n_fmaps, init=init, scope='pixel_cnn')

            if self.dataset == "mnist":
                logits = tf.reshape(z, shape=[-1, self.n_x])
                probs = tf.nn.sigmoid(logits)

            elif self.dataset == "color":
                logits = tf.reshape(z, shape=[-1, self.n_x, n_cats])
                probs = tf.nn.softmax(logits, dim=-1)
            else:
                raise NotImplementedError


            return logits, probs


    def _reconstruction(self, logits, labels, scope):

        with tf.variable_scope(scope):

            if self.dataset == "mnist":
                l1 = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), axis=1)
                l1 = tf.reduce_mean(l1, axis=0)

            elif self.dataset == "color":
                labels = tf.cast(labels * 255, dtype=tf.int32)   # integers [0,255] inclusive
                l1 = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                l1 = tf.reduce_sum(l1, axis=1)
                l1 = tf.reduce_mean(l1, axis=0)

            else:
                raise NotImplementedError

            return l1


    def _penalty(self, mu, sigma, log_q, z_K, scope):

        with tf.variable_scope(scope):

            if self.is_flow:
                D = z_K.get_shape()[1].value

                log_p = -0.5 * tf.square(z_K)  - (D / 2) * np.log(2*np.pi)

                penalty = tf.reduce_sum(-log_q + log_p, axis=1)
                penalty = tf.reduce_mean(penalty, axis=0)

            else:

                D = self.n_z

                log_p = -0.5*(tf.square(mu) + tf.square(sigma)) #- (D / 2)*np.log(2*np.pi)
                log_q = -0.5*(1 + 2*tf.log(sigma)) #- (D / 2)*np.log(2*np.pi)

                penalty = 0.5 * tf.reduce_sum(1 + 2*tf.log(sigma) - tf.square(mu) - tf.square(sigma), axis=1)
                penalty = tf.reduce_mean(penalty, axis=0)

            return penalty, log_q, log_p



    def _variational_bound(self, scope):

        with tf.variable_scope(scope):
            return self.l1 + self.l2


    def _loss(self, scope):

        with tf.variable_scope(scope):
            alpha = self.args['anneal']

            if alpha < 0:
                # free bits penalty  (also works with normalizing flows)
                l2 = tf.reduce_mean(-self.log_q + self.log_p, axis=0)
                l2 = tf.minimum(l2, alpha)
                l2 = tf.reduce_sum(l2, axis=1)

            else:
                l2 = self.l2

            return -(self.l1 + l2)


    def _optimizer(self, loss, scope='optimizer'):

        with tf.variable_scope(scope):
            lr = self.args['learning_rate']
            step = tf.train.RMSPropOptimizer(lr).minimize(loss)

            return step


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


    def _summaries(self,):

        with tf.variable_scope("summaries"):
            tf.summary.scalar('lower_bound', self.bound)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('reconstruction', self.l1)
            tf.summary.scalar('penalty', self.l2)

            tf.summary.scalar('sigma0', tf.reduce_mean(self.z_sigma))

            if self.is_flow:
                c = - (self.n_z / 2)*np.log(2*np.pi)

                lq = tf.reduce_sum(self.log_q - c, axis=1)
                tf.summary.scalar('penalty_log_q', tf.reduce_mean(lq, axis=0))

                lp = tf.reduce_sum(self.log_p - c, axis=1)
                tf.summary.scalar('penalty_log_p', tf.reduce_mean(lp, axis=0))

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
        if self.is_autoregressive:
            n_pixels = self.args['n_conditional_pixels']
            z = self.encode(x, mean=False)
            return self._autoregressive_sampling(z, x, n_pixels)

        else:
            feed = {self.x: x, self.is_training: False}
            return self.sess.run(self.rx_probs, feed_dict=feed)


    def encode(self, x, mean=False):
        """
        Encode x.
        """
        feed = {self.x: x, self.is_training: False}
        if mean:
            assert self.is_flow == False
            return self.sess.run(self.z_mu, feed_dict=feed)
        else:
            return self.sess.run(self.z, feed_dict=feed)


    def decode(self, z):
        """
        Decodes z.
        """
        if self.is_autoregressive:
            x = np.random.rand(z.shape[0], self.n_x)
            return self._autoregressive_sampling(z, x, n_pixels=0)

        else:
            feed = {self.z: z, self.is_training: False}
            return self.sess.run(self.rx_probs, feed_dict=feed)


    def sample_prior(self, n_samples):
        """
        Samples z from prior distribution.
        """
        return np.random.normal(size=[n_samples, self.n_z])






