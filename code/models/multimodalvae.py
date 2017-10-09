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
        self.embed_size = self.args['embed_size']           # embedding size
        self.gru_layers = self.args['gru_layers']           # number of GRU layers
        self.n_units = self.args['n_units']                 # number of hidden units in FC layers
        self.n_fmaps = self.args['n_feature_maps']          # number of feature maps in Conv. layers
        self.alpha = self.args['anneal']                    # freebits parameter
        self.joint_anneal = self.args['joint_anneal']       # joint annealing parameter
        self.lr = self.args['learning_rate']                # learning rate

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
        print("Placeholders...", flush=True)
        self.xi = tf.placeholder(tf.float32, [None, self.nxi], name='xi')
        self.xc = tf.placeholder(tf.int32, [None, self.nxc], name='xc')
        self.xpi = tf.placeholder(tf.float32, [None, self.nxi], name='xpi')
        self.xpc = tf.placeholder(tf.int32, [None, self.nxc], name='xpc')

        # data-dependent weight initialization (Salisman, Kingma - 2016)
        print("Sample batch...", flush=True)
        xi_init = tf.constant(self.xi_init, tf.float32)
        xc_init = tf.constant(self.xc_init, tf.int32)
        xpi_init = tf.constant(self.xpi_init, tf.float32)
        xpc_init = tf.constant(self.xpc_init, tf.int32)


        # compute weight initializations
        print("Initialize model weights...", flush=True)
        self._model((xi_init, xc_init, xpi_init, xpc_init), init=True)

        # model specification
        print("Define model connections...", flush=True)
        self._model((self.xi, self.xc, self.xpi, self.xpc), init=False)


        # marginal bounds
        print("Marginal bounds...", flush=True)
        self.lxi, self.lxirec, self.lxipen, self.logqi, self.logpi = self._marginal_bound(self.rxi_i, self.xi,
                                        self.mu_i, self.sigma_i, dtype='image', scope='marg_xi')

        self.lxc, self.lxcrec, self.lxcpen, self.logqc, self.logpc = self._marginal_bound(self.rxc_c, self.xc,
                                        self.mu_c, self.sigma_c, dtype='caption', scope='marg_xc')

        self.lxpi, self.lxpirec, self.lxpipen, self.logqpi, self.logppi = self._marginal_bound(self.rxi_pi, self.xpi,
                             self.mu_pi, self.sigma_pi, dtype='image', scope='marg_xpi')

        self.lxpc, self.lxpcrec, self.lxpcpen, self.logqpc, self.logppc = self._marginal_bound(self.rxc_pc, self.xpc,
                             self.mu_pc, self.sigma_pc, dtype='caption', scope='marg_xpc')


        # joint bound
        print("Joint bound...", flush=True)
        self.lxj, self.lxjreci, self.lxjrecc, self.lxjpen, self.logqj, self.logpj = self._joint_bound(
            self.rxi_j, self.xpi, self.rxc_j, self.xpc, self.mu_j, self.sigma_j, scope='joint_bound')

        # translation bounds
        print("Translation bounds...", flush=True)
        self.txi = self._translation_bound(self.rxi_pc, self.xpi, dtype='image', scope='translate_to_xi')
        self.txc = self._translation_bound(self.rxc_pi, self.xpc, dtype='caption', scope='translate_to_xc')

        # loss function
        print("Loss function...", flush=True)
        self.loss = self._loss(scope='loss')

        # optimizer
        print("Optimizer...", flush=True)
        self.step = self._optimizer(self.loss, scope='optimizer')

        # summary variables
        print("Summary variables...", flush=True)
        self.summary = self._summaries()



    def _model(self, xs, init):

        with tf.variable_scope('multimodal_vae') as scope:
            if not init:
                scope.reuse_variables()

            xi, xc, xpi, xpc = xs

            # encoders
            print("Encoders...", flush=True)
            mu_pi, sigma_pi, hepi = self._encoder_i(xpi, init=init, scope='xi_enc')
            mu_pc, sigma_pc, hepc, emb_pc = self._encoder_c(xpc, init=init, scope='xc_enc')

            # joint encoder
            print("Joint Encoder...", flush=True)
            mu_j, sigma_j = self._joint_encoder(xhi=hepi, xhc=hepc, init=init, scope='xixc_enc')

            # sample network
            print("Sampler...", flush=True)
            zj = self._sample(mu_j, sigma_j, scope='sample')

            # decoders
            print("Decoders...", flush=True)
            rxi_j, rxi_j_probs = self._decoder_i(zj, init=init, scope='xi_dec')
            rxc_j, rxc_j_probs = self._decoder_c(zj, emb_pc, init=init, scope='xc_dec')


            if not init:

                # unpaired encodings
                print("Encoders (2)...", flush=True)
                mu_i, sigma_i, _ = self._encoder_i(xi, init=init, scope='xi_enc')
                mu_c, sigma_c, _, emb_c = self._encoder_c(xc, init=init, scope='xc_enc')

                # additional samples
                print("Samplers (2)...", flush=True)
                zi = self._sample(mu_i, sigma_i, scope='sample')
                zc = self._sample(mu_c, sigma_c, scope='sample')
                zpi = self._sample(mu_pi, sigma_pi, scope='sample')
                zpc = self._sample(mu_pc, sigma_pc, scope='sample')

                # reconstructions
                print("Decoders (2)...", flush=True)
                rxi_i, rxi_i_probs = self._decoder_i(zi, init=init, scope='xi_dec')
                rxc_c, rxc_c_probs = self._decoder_c(zc, emb_c, init=init, scope='xc_dec')

                # translations
                rxi_c, rxi_c_probs = self._decoder_i(zc, init=init, scope='xi_dec')
                rxc_i, rxc_i_probs = self._decoder_c(zi, emb_c, init=init, scope='xc_dec')

                # reconstructions (from paired input)
                rxi_pi, rxi_pi_probs = self._decoder_i(zpi, init=init, scope='xi_dec')
                rxc_pc, rxc_pc_probs = self._decoder_c(zpc, emb_pc, init=init, scope='xc_dec')

                # translations (from paired input)
                rxi_pc, rxi_pc_probs = self._decoder_i(zpc, init=init, scope='xi_dec')
                rxc_pi, rxc_pi_probs = self._decoder_c(zpi, emb_pc, init=init, scope='xc_dec')

                # final tensors
                print("Assignments...", flush=True)
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

        mu, sigma, he, emb = nw.seq_encoder(x,  self.vocab_size, self.embed_size, self.n_units,
                                       self.n_z, self.gru_layers, init, scope)

        return mu, sigma, he, emb



    def _sample(self, mu0, sigma0, scope):

        with tf.variable_scope(scope):
            n_samples = tf.shape(mu0)[0]
            epsilon = tf.random_normal((n_samples, self.n_z))

            return mu0 + tf.multiply(sigma0, epsilon)


    def _decoder_i(self, z, init, scope):

        nch = self.nch * 256

        z = nw.deconvolution_coco(z, nch, self.n_fmaps, self.n_units, init, scope)

        logits = tf.reshape(z, shape=[-1, self.nxi, 256])
        parms = tf.nn.softmax(logits, dim=-1)

        return logits, parms


    def _decoder_c(self, z, x_emb, init, scope):

        z = nw.seq_decoder(z, x_emb, self.vocab_size, self.embed_size, self.gru_layers, init, scope)

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


    def _translation_bound(self, logits, labels, dtype, scope):

        with tf.variable_scope(scope):
            return self._reconstruction(logits=logits, labels=labels, dtype=dtype, scope='reconstruction')


    def _optimizer(self, loss, scope):

        with tf.variable_scope(scope):
            step = tf.train.RMSPropOptimizer(self.lr).minimize(loss)

            return step


    def _freebits(self, l2, log_q, log_p, alpha):
        """
        Compute freebits penalty if alpha < 0, otherwise return original penalty l2.
        """
        if alpha < 0:
            l2 = tf.reduce_mean(-log_q + log_p, axis=0)
            l2 = tf.minimum(l2, alpha)
            l2 = tf.reduce_sum(l2)

        return l2


    def _loss(self, scope):

        with tf.variable_scope(scope):

            # marginal x1
            lxipen = self._freebits(self.lxipen, self.logqi, self.logpi, self.alpha)
            lxi = self.lxirec + lxipen

            # marginal x2
            lxcpen = self._freebits(self.lxcpen, self.logqc, self.logpc, self.alpha)
            lxc = self.lxcrec + lxcpen

            if self.objective == "joint":
                # joint xi and xc
                lxjpen = self._freebits(self.lxjpen, self.logqj, self.logpj, self.alpha)
                lxj = self.lxjreci + self.lxjrecc + lxjpen

                bound = (self.joint_anneal * lxj) + lxi + lxc

            elif self.objective == "translate":
                # marginal x1p
                lxpipen = self._freebits(self.lxpipen,  self.logqpi, self.logppi, self.alpha)
                lxpi = self.lxpirec + lxpipen

                # marginal x2p
                lxpcpen = self._freebits(self.lxpcpen, self.logqpc, self.logppc, self.alpha)
                lxpc = self.lxpcrec + lxpcpen

                bound = (self.txi + lxpc) + (self.txc + lxpi) + lxi + lxc

            else:
                raise NotImplementedError

            loss = -bound
            return loss


    def _summaries(self,):

        with tf.variable_scope("summaries"):
            tf.summary.scalar('loss_(ignore_test)', self.loss)

            if self.objective == "joint":
                tf.summary.scalar('lower_bound_on_log_p_xi_xc', self.lxj)

            elif self.objective == "translate":
                tf.summary.scalar('lower_bound_on_log_p_xi_xc_ti', self.txi + self.lxpc)
                tf.summary.scalar('lower_bound_on_log_p_xi_xc_tc', self.txc + self.lxpi)

            else:
                raise NotImplementedError

            tf.summary.scalar('marg_xi_(ignore_test)', self.lxi)
            tf.summary.scalar('marg_xc_(ignore_test)', self.lxc)
            tf.summary.scalar('marg_xpi', self.lxpi)
            tf.summary.scalar('marg_xpc', self.lxpc)

            tf.summary.scalar('joint', self.lxj)
            tf.summary.scalar('trans_to_xi', self.txi)
            tf.summary.scalar('trans_to_xc', self.txc)

            return tf.summary.merge_all()


    def _track(self, terms, prefix):

        if self.tracker is not None:

            for name, term in terms.items():
                self.tracker.add(i=self.n_steps, value=term, series_name=prefix+name, run_name=self.name)


    def train(self, xs):
        """
        Performs single training step.
        """
        xi, xc, xi_pairs, xc_pairs = xs

        feed = {self.xi: xi, self.xc: xc, self.xpi: xi_pairs, self.xpc: xc_pairs}
        outputs = [self.summary, self.step, self.loss, self.lxi, self.lxc, self.lxj, self.txi, self.txc, self.lxpi, self.lxpc]

        summary, _, loss, lxi, lxc, lxj, txi, txc, lxpi, lxpc = self.sess.run(outputs, feed_dict=feed)

        if self.objective == 'joint':
            bound = lxj
            terms = {'lower_bound_on_log_p_xi_xc': bound, 'loss': loss,
                     'lxi': lxi, 'lxc': lxc, 'lxj': lxj, 'txi': txi, 'txc': txc}
        else:  # translate
            bound_ti = txi + lxpc
            bound_tc = txc + lxpi
            terms = {'lower_bound_on_log_p_xi_xc_ti': bound_ti, 'lower_bound_on_log_p_xi_xc_tc': bound_tc, 'loss': loss,
                     'lxi': lxi, 'lxc': lxc, 'lxj': lxj, 'txi': txi, 'txc': txc}

        self._track(terms, prefix='train_')
        self.tr_writer.add_summary(summary, self.n_steps)

        self.n_steps = self.n_steps + 1


    def test(self, xs):
        """
        Computes lower bound on test data.
        """
        xi, xc = xs

        feed = {self.xi: xi, self.xc: xc, self.xpi: xi, self.xpc: xc}
        outputs = [self.summary, self.loss, self.lxpi, self.lxpc, self.lxj, self.txi, self.txc, self.lxpi, self.lxpc]

        summary, loss, lxi, lxc, lxj, txi, txc, lxpi, lxpc = self.sess.run(outputs, feed_dict=feed)

        if self.objective == 'joint':
            bound = lxj
            terms = {'lower_bound_on_log_p_xi_xc': bound, 'loss': loss,
                     'lxi': lxi, 'lxc': lxc, 'lxj': lxj, 'txi': txi, 'txc': txc}

        else:  # translate
            bound_ti = txi + lxpc
            bound_tc = txc + lxpi
            terms = {'lower_bound_on_log_p_xi_xc_ti': bound_ti, 'lower_bound_on_log_p_xi_xc_tc': bound_tc, 'loss': loss,
                     'lxi': lxi, 'lxc': lxc, 'lxj': lxj, 'txi': txi, 'txc': txc}

        self._track(terms, prefix='test_')
        self.te_writer.add_summary(summary, self.n_steps)


    def reconstruct(self, xs):
        """
        Reconstruct input.
        If one input is None, then this can be interpretted as a translation from the other input.

        xs: (xi, xc) or (xi, None) or (None, xc)
        """
        z = self.encode(xs, mean=False)
        xi, xc = self.decode(z)

        return xi, xc


    def encode(self, xs, mean=False):
        """
        Encodes xi or xc or both xi and xc to the latent space.
        """
        xi, xc = xs
        feed = {self.xpi: xi, self.xpc: xc}
        outputs = self.mu_j, self.zj

        if xi is None and xc is None:
            raise ValueError

        elif xi is None:
            feed = {self.xpc: xc}
            outputs = self.mu_pc, self.zpc

        elif xc is None:
            feed = {self.xpi: xi}
            outputs = self.mu_pi, self.zpi

        out = outputs[0] if mean else outputs[1]

        return self.sess.run(out, feed_dict=feed)


    def decode(self, z):

        # autoregressive caption generation
        xc = np.random.randint(0, self.vocab_size, size=(z.shape[0], self.nxc))
        xc = self._autoregressive_sampling(z, xc)

        # factorized image generation
        feed = {self.zj: z}
        rxi = self.sess.run(self.rxi_j_probs, feed_dict=feed)
        xi = self._categorical_sampling(rxi, n_cats=256) / 255

        return xi, xc


    def _categorical_sampling(self, rx, n_cats):
        """
        Categorical sampling. Probabilities assumed to be on third dimension of three dimensional vector.
        """
        batch = rx.shape[0]
        features = rx.shape[1]
        rxp = np.empty([batch, features], dtype=rx.dtype)

        for i in range(batch):
            for j in range(features):
                rxp[i, j] = np.random.choice(a=n_cats, p=rx[i, j])

        return rxp


    def _autoregressive_sampling(self, z, xc):
        """
        Synthesize captions autoregressively.
        """
        max_seq_len = self.nxc

        for i in range(max_seq_len):

            feed = {self.zj: z, self.xpc: xc}
            pb = self.sess.run(self.rxc_j_probs, feed_dict=feed)   # batch_size x max_seq_len x vocab_size

            pb = pb[:,i,:]
            pb = np.expand_dims(pb, axis=1)

            pb = self._categorical_sampling(pb, n_cats=self.vocab_size)
            pb = np.squeeze(pb)

            xc[:,i] = pb

        return xc
