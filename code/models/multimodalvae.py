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
        self.xc = tf.placeholder(tf.int32, [None, self.nxc], name='xc')
        self.xpi = tf.placeholder(tf.float32, [None, self.nxi], name='xpi')
        self.xpc = tf.placeholder(tf.int32, [None, self.nxc], name='xpc')

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

        self.lx1, self.lx1rec, self.lx1pen, self.logq1, self.logp1 = self._marginal_bound(self.rxi_i, self.xi,
                                        self.mu_i, self.sigma_i, dtype='image', scope='marg_xi')

        self.lx2, self.lx2rec, self.lx2pen, self.logq2, self.logp2 = self._marginal_bound(self.rxc_c, self.xc,
                                        self.mu_c, self.sigma_c, dtype='caption', scope='marg_xc')

        self.lx1p, self.lx1prec, self.lx1ppen, self.logq1p, self.logp1p = self._marginal_bound(self.rxi_pi, self.xpi,
                             self.mu_pi, self.sigma_pi, dtype='image', scope='marg_xpi')

        self.lx2p, self.lx2prec, self.lx2ppen, self.logq2p, self.logp2p = self._marginal_bound(self.rxc_pc, self.xpc,
                             self.mu_pc, self.sigma_pc, dtype='caption', scope='marg_xpc')


        # joint bound

        self.lx12, self.lx12rec1, self.lx12rec2, self.lx12pen, self.logq12, self.logp12 = self._joint_bound(
            self.rxi_j, self.xpi, self.rxc_j, self.xpc, self.mu_j, self.sigma_j, scope='joint_bound')

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

            xi, xc, xpi, xpc = xs

            # encoders
            mu_pi, sigma_pi, hepi = self._encoder_i(xpi, init=init, scope='xi_enc')
            mu_pc, sigma_pc, hepc = self._encoder_c(xpc, init=init, scope='xc_enc')

            # joint encoder
            mu_j, sigma_j = self._joint_encoder(xhi=hepi, xhc=hepc, init=init, scope='xixc_enc')

            # sample network
            zj = self._sample(mu_j, sigma_j, scope='sample')

            # decoders
            rxi_j, rxi_j_probs = self._decoder_i(zj, init=init, scope='xi_dec')
            rxc_j, rxc_j_probs = self._decoder_c(zj, init=init, scope='xc_dec')

        if not init:
            with tf.variable_scope('multimodal_vae') as scope:
                scope.reuse_variables()

                # unpaired encodings
                mu_i, sigma_i, _ = self._encoder_i(xi, init=init, scope='xi_enc')
                mu_c, sigma_c, _ = self._encoder_c(xc, init=init, scope='xc_enc')

                # additional samples
                zi = self._sample(mu_i, sigma_i, scope='sample')
                zc = self._sample(mu_c, sigma_c, scope='sample')
                zpi = self._sample(mu_pi, sigma_pi, scope='sample')
                zpc = self._sample(mu_pc, sigma_pc, scope='sample')

                # reconstructions
                rxi_i, rxi_i_probs = self._decoder_i(zi, init=init, scope='xi_dec')
                rxc_c, rxc_c_probs = self._decoder_c(zc, init=init, scope='xc_dec')

                # translations
                rxi_c, rxi_c_probs = self._decoder_i(zc, init=init, scope='xi_dec')
                rxc_i, rxc_i_probs = self._decoder_c(zi, init=init, scope='xc_dec')

                # reconstructions (from paired input)
                rxi_pi, rxi_pi_probs = self._decoder_i(zpi, init=init, scope='xi_dec')
                rxc_pc, rxc_pc_probs = self._decoder_c(zpc, init=init, scope='xc_dec')

                # translations (from paired input)
                rxi_pc, rxi_pc_probs = self._decoder_i(zpc, init=init, scope='xi_dec')
                rxc_pi, rxc_pi_probs = self._decoder_c(zpi, init=init, scope='xc_dec')

                # final tensors
                self.mu_j, self.sigma_j = (mu_j, sigma_j)
                self.mu_i, self.sigma_i = (mu_i, sigma_i)
                self.mu_c, self.sigma_c = (mu_c, sigma_c)
                self.mu_pi, self.sigma_pi = (mu_pi, sigma_pi)
                self.mu_pc, self.sigma_pc = (mu_pc, sigma_pc)

                self.zj, self.zi, self.zc, self.zpi, self.zpc = (zj, zi, zc, zpi, zpc)

                self.rxi_j, self.rxi_j_probs = (rxi_j, rxi_j_probs)
                self.rxc_j, self.rxc_j_probs = (rxc_j, rxc_j_probs)

                self.rxi_i, self.rxi_i_probs = (rxi_i, rxi_i_probs)
                self.rxc_c, self.rxc_c_probs = (rxc_c, rxc_c_probs)
                self.rxi_c, self.rxi_c_probs = (rxi_c, rxi_c_probs)
                self.rxc_i, self.rxc_i_probs = (rxc_i, rxc_i_probs)

                self.rxi_pi, self.rxi_pi_probs = (rxi_pi, rxi_pi_probs)
                self.rxc_pc, self.rxc_pc_probs = (rxc_pc, rxc_pc_probs)
                self.rxi_pc, self.rxi_pc_probs = (rxi_pc, rxi_pc_probs)
                self.rxc_pi, self.rxc_pi_probs = (rxc_pi, rxc_pi_probs)


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


    def _joint_encoder(self, xhi, xhc, init, scope):

        with tf.variable_scope(scope):

            z_mu, z_sigma = nw.joint_coco_encode(xhi, xhc, self.n_units, self.n_z, init, scope=scope)

            return z_mu, z_sigma


    def _reconstruction(self, logits, labels, dtype, scope):

        with tf.variable_scope(scope):

            if dtype == 'image':

                labels = tf.cast(labels * 255, dtype=tf.int32)  # integers [0,255] inclusive
                l1 = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

                l1 = tf.reduce_sum(l1, axis=1)
                l1 = tf.reduce_mean(l1, axis=0)

                return l1

            elif dtype == 'caption':

                l1 = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

                l1 = tf.reduce_sum(l1, axis=1)
                l1 = tf.reduce_mean(l1, axis=0)

                return l1

            else:
                raise NotImplementedError


    def _penalty(self, mu, sigma, scope):

        with tf.variable_scope(scope):

            log_p = -0.5*(tf.square(mu) + tf.square(sigma)) #- 0.5*np.log(2*np.pi)
            log_q = -0.5*(1 + 2*tf.log(sigma)) #- 0.5*np.log(2*np.pi)

            penalty = 0.5 * tf.reduce_sum(1 + 2*tf.log(sigma) - tf.square(mu) - tf.square(sigma), axis=1)
            penalty = tf.reduce_mean(penalty, axis=0)

            return penalty, log_q, log_p


    def _marginal_bound(self, logits, labels, mu, sigma, dtype, scope):

        with tf.variable_scope(scope):

            l1 = self._reconstruction(logits=logits, labels=labels, dtype=dtype, scope='reconstruction')
            l2, log_q, log_p = self._penalty(mu=mu, sigma=sigma, scope='penalty')
            bound = l1 + l2

            return bound, l1, l2, log_q, log_p


    def _joint_bound(self, xi_logits, xi_labels, xc_logits, xc_labels, mu, sigma, scope):

        with tf.variable_scope(scope):

            l1_xi = self._reconstruction(logits=xi_logits, labels=xi_labels, dtype='image', scope='reconstruction_xi')
            l1_xc = self._reconstruction(logits=xc_logits, labels=xc_labels, dtype='caption', scope='reconstruction_xc')

            l2, log_q, log_p = self._penalty(mu=mu, sigma=sigma, scope='penalty')

            bound = l1_xi + l1_xc + l2

            return bound, l1_xi, l1_xc, l2, log_q, log_p


    def _translation_bound(self, logits, labels, scope):

        with tf.variable_scope(scope):
            return self._reconstruction(logits=logits, labels=labels, scope='reconstruction')