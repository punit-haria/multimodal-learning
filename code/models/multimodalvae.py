import tensorflow as tf
import numpy as np
from copy import deepcopy

from models import base
from models import layers as nw


class MultiModalVAE(base.Model):
    """
    Image-text multimodal variational autoencoder.
    """
    def __init__(self, arguments, name, tracker, init_minibatches, session=None, log_dir=None, model_dir=None):

        self.args = deepcopy(arguments)

        # options
        self.n_z = self.args['n_z']                         # latent dimensionality
        self.objective = self.args['objective']             # joint vs. translation objective function
        self.nxc = self.args['max_seq_len']                 # maximum caption length
        self.vocab_size = self.args['vocab_size']           # vocabulary size
        self.n_units = self.args['n_units']                 # number of hidden units in FC layers
        self.n_fmaps = self.args['n_feature_maps']          # number of feature maps in Conv. layers

        # image dimensions
        self.h, self.w, self.nch = (48, 64, 3)
        self.nxi = self.h * self.w * self.nch

        # object to track model performance (can be None)
        self.tracker = tracker
        if self.tracker is not None:
            self.tracker.create_run(run_name=name, model_name=self.__class__.__name__, parameters=self.args)

        # training steps counter
        self.n_steps = 0

        # initialization minibatches
        self.xi_init, self.xc_init, self.xpi_init, self.xpc_init = init_minibatches

        # constructor
        super(MultiModalVAE, self).__init__(name=name, session=session, log_dir=log_dir, model_dir=model_dir)


    def _initialize(self,):

        # input placeholders
        self.xi = tf.placeholder(tf.float32, [None, self.nxi], name='xi')
        self.xc = tf.placeholder(tf.float32, [None, self.nxc], name='xc')
        self.xpi = tf.placeholder(tf.float32, [None, self.nxi], name='xpi')
        self.xpc = tf.placeholder(tf.float32, [None, self.nxc], name='xpc')

        # data-dependent weight initialization (Salisman, Kingma - 2016)
        xi_init = tf.constant(self.xi_init, tf.float32)
        xc_init = tf.constant(self.xc_init, tf.float32)
        xpi_init = tf.constant(self.xpi_init, tf.float32)
        xpc_init = tf.constant(self.xpc_init, tf.float32)


        # compute weight initializations
        self._model((xi_init, xc_init, xpi_init, xpc_init), init=True)

        # model specification
        self._model((self.xi, self.xc, self.xpi, self.xpc), init=False)


        # marginal bounds

        self.lx1, self.lx1rec, self.lx1pen, self.logq1, self.logp1 = self._marginal_bound(self.rx1_1, self.x1,
                                        self.z1_mu, self.z1_sigma, self.log_q1, self.z1, scope='marg_x1')

        self.lx2, self.lx2rec, self.lx2pen, self.logq2, self.logp2 = self._marginal_bound(self.rx2_2, self.x2,
                                        self.z2_mu, self.z2_sigma, self.log_q2, self.z2, scope='marg_x2')

        self.lx1p, self.lx1prec, self.lx1ppen, self.logq1p, self.logp1p = self._marginal_bound(self.rx1_1p, self.x1p,
                             self.z1p_mu, self.z1p_sigma, self.log_q1p, self.z1p, scope='marg_x1p')

        self.lx2p, self.lx2prec, self.lx2ppen, self.logq2p, self.logp2p = self._marginal_bound(self.rx2_2p, self.x2p,
                             self.z2p_mu, self.z2p_sigma, self.log_q2p, self.z2p, scope='marg_x2p')


        # joint bound

        self.lx12, self.lx12rec1, self.lx12rec2, self.lx12pen, self.logq12, self.logp12 = self._joint_bound(
            self.rx1_12, self.xpi, self.rx2_12, self.xpc,
            self.z12_mu, self.z12_sigma, self.log_q12, self.z12, scope='joint_bound')

        # translation bounds

        self.tx1 = self._translation_bound(self.rx1_2p, self.xpi, scope='translate_to_x1')
        self.tx2 = self._translation_bound(self.rx2_1p, self.xpc, scope='translate_to_x2')

        # loss function
        self.loss = self._loss(scope='loss')

        # optimizer
        self.step = self._optimizer(self.loss)

        # summary variables
        self.summary = self._summaries()



    def _model(self, xs, init):

        with tf.variable_scope('multimodal_vae') as scope:
            if not init:
                scope.reuse_variables()

            extra = self.args['flow']

            xi, xc, xpi, xpc = xs

            # encoders
            z1p_mu, z1p_sigma, h1p, he1p = self._encoder(x1p, init=init, scope='x1_enc')
            z2p_mu, z2p_sigma, h2p, he2p = self._encoder(x2p, init=init, scope='x2_enc')

            # joint encoder
            z12_mu, z12_sigma, h12 = self._joint_encoder(z1p_mu, z1p_sigma, h1p, z2p_mu, z2p_sigma, h2p,
                                                         x1, x2, x1h=he1p, x2h=he2p,
                                                         extra=extra, init=init, scope='x1x2_enc')

            # sample network
            z12, log_q12 = self._sample(z12_mu, z12_sigma, h12, scope='sample')

            # decoders
            rx1_12, rx1_12_probs = self._decoder(z12, x1p, init=init, scope='x1_dec')
            rx2_12, rx2_12_probs = self._decoder(z12, x2p, init=init, scope='x2_dec')

        if not init:
            with tf.variable_scope('multimodal_vae') as scope:
                scope.reuse_variables()

                # unpaired encodings
                z1_mu, z1_sigma, h1, _ = self._encoder(x1, init=init, scope='x1_enc')
                z2_mu, z2_sigma, h2, _ = self._encoder(x2, init=init, scope='x2_enc')

                # additional samples
                z1, log_q1 = self._sample(z1_mu, z1_sigma, h1, scope='sample')
                z2, log_q2 = self._sample(z2_mu, z2_sigma, h2, scope='sample')
                z1p, log_q1p = self._sample(z1p_mu, z1p_sigma, h1p, scope='sample')
                z2p, log_q2p = self._sample(z2p_mu, z2p_sigma, h2p, scope='sample')

                # reconstructions
                rx1_1, rx1_1_probs = self._decoder(z1, x1, init=init, scope='x1_dec')
                rx2_2, rx2_2_probs = self._decoder(z2, x2, init=init, scope='x2_dec')

                # translations
                rx1_2, rx1_2_probs = self._decoder(z2, x1, init=init, scope='x1_dec')
                rx2_1, rx2_1_probs = self._decoder(z1, x2, init=init, scope='x2_dec')

                # reconstructions (from paired input)
                rx1_1p, rx1_1p_probs = self._decoder(z1p, x1p, init=init, scope='x1_dec')
                rx2_2p, rx2_2p_probs = self._decoder(z2p, x2p, init=init, scope='x2_dec')

                # translations (from paired input)
                rx1_2p, rx1_2p_probs = self._decoder(z2p, x1p, init=init, scope='x1_dec')
                rx2_1p, rx2_1p_probs = self._decoder(z1p, x2p, init=init, scope='x2_dec')

                # final tensors
                self.z12_mu, self.z12_sigma = (z12_mu, z12_sigma)
                self.z1_mu, self.z1_sigma = (z1_mu, z1_sigma)
                self.z2_mu, self.z2_sigma = (z2_mu, z2_sigma)
                self.z1p_mu, self.z1p_sigma = (z1p_mu, z1p_sigma)
                self.z2p_mu, self.z2p_sigma = (z2p_mu, z2p_sigma)

                self.z12, self.log_q12 = (z12, log_q12)
                self.z1, self.log_q1 = (z1, log_q1)
                self.z2, self.log_q2 = (z2, log_q2)
                self.z1p, self.log_q1p = (z1p, log_q1p)
                self.z2p, self.log_q2p = (z2p, log_q2p)

                self.rx1_12, self.rx1_12_probs = (rx1_12, rx1_12_probs)
                self.rx2_12, self.rx2_12_probs = (rx2_12, rx2_12_probs)

                self.rx1_1, self.rx1_1_probs = (rx1_1, rx1_1_probs)
                self.rx2_2, self.rx2_2_probs = (rx2_2, rx2_2_probs)
                self.rx1_2, self.rx1_2_probs = (rx1_2, rx1_2_probs)
                self.rx2_1, self.rx2_1_probs = (rx2_1, rx2_1_probs)

                self.rx1_1p, self.rx1_1p_probs = (rx1_1p, rx1_1p_probs)
                self.rx2_2p, self.rx2_2p_probs = (rx2_2p, rx2_2p_probs)
                self.rx1_2p, self.rx1_2p_probs = (rx1_2p, rx1_2p_probs)
                self.rx2_1p, self.rx2_1p_probs = (rx2_1p, rx2_1p_probs)


    def _encoder_i(self, x, init, scope):

        mu, sigma, he = nw.convolution_coco(x, self.nch, self.n_fmaps, self.n_units, self.n_z, init, scope)

        return mu, sigma, he


    def _encoder_c(self, x, init, scope):

        mu, sigma, he = nw.seq_encoder(x, self.nch, self.n_units, self.n_z, init, scope)

        return mu, sigma, he



    def _sample(self, mu0, sigma0, scope):

        with tf.variable_scope(scope):
            n_samples = tf.shape(mu0)[0]
            epsilon = tf.random_normal((n_samples, self.n_z))

            return mu0 + tf.multiply(sigma0, epsilon)


    def _decoder_i(self, z, init, scope):

        with tf.variable_scope(scope):

            nch = self.nch * 256

            z = nw.deconvolution_coco(z, nch, self.n_fmaps, self.n_units, init, scope)

            logits = tf.reshape(z, shape=[-1, self.nxi, 256])
            parms = tf.nn.softmax(logits, dim=-1)

            return logits, parms


    def _decoder_c(self, z, init, scope):

        with tf.variable_scope(scope):

            nxc = self.nxc * self.vocab_size

            z = nw.seq_decoder(z, nxc, self.n_units, init, scope)

            logits = tf.reshape(z, shape=[-1, self.nxc, self.vocab_size])
            parms = tf.nn.softmax(logits, dim=-1)

            return logits, parms
