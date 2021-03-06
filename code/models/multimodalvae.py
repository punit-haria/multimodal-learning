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
        self.n_z = self.args['n_z']                             # latent dimensionality
        self.objective = self.args['objective']                 # joint vs. translation objective function
        self.nxc = self.args['max_seq_len']                     # maximum caption length
        self.vocab_size = self.args['vocab_size']               # vocabulary size
        self.embed_size = self.args['embed_size']               # embedding size
        self.gru_layers = self.args['gru_layers']               # number of GRU layers
        self.n_units_image = self.args['n_units_image']         # number of hidden units in FC layers
        self.n_units_enc_capt = self.args['n_units_enc_capt']   # number of hidden units in Encoder GRU
        self.n_fmaps_capt = self.args['n_feature_maps_capt']    # number of feature maps in Caption Decoder
        self.n_fmaps_image = self.args['n_feature_maps_image']  # number of feature maps in Image Decoder
        self.alpha = self.args['anneal']                        # freebits parameter
        self.joint_anneal = self.args['joint_anneal']           # joint annealing parameter
        self.lr = self.args['learning_rate']                    # learning rate
        self.softmax_samples = self.args['softmax_samples']     # softmax samples

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
        self.xi_init, self.xc_init, self.sl_init, self.xc_dec_init, \
        self.xpi_init, self.xpc_init, self.slp_init, self.xpc_dec_init = init_minibatches

        # constructor
        super(MultiModalVAE, self).__init__(name=name, session=session, log_dir=log_dir, model_dir=model_dir)


    def _initialize(self,):

        # input placeholders
        print("Placeholders...", flush=True)
        self.xi = tf.placeholder(tf.float32, [None, self.nxi], name='xi')
        self.xc = tf.placeholder(tf.int32, [None, self.nxc], name='xc')
        self.sl = tf.placeholder(tf.int32, [None], name='sl')
        self.xc_dec = tf.placeholder(tf.int32, [None, self.nxc], name='xc_dec')

        self.xpi = tf.placeholder(tf.float32, [None, self.nxi], name='xpi')
        self.xpc = tf.placeholder(tf.int32, [None, self.nxc], name='xpc')
        self.slp = tf.placeholder(tf.int32, [None], name='slp')
        self.xpc_dec = tf.placeholder(tf.int32, [None, self.nxc], name='xpc_dec')

        # data-dependent weight initialization (Salisman, Kingma - 2016)
        print("Sample batch...", flush=True)
        xi_init = tf.constant(self.xi_init, tf.float32)
        xc_init = tf.constant(self.xc_init, tf.int32)
        sl_init =  tf.constant(self.sl_init, tf.int32)
        xc_dec_init = tf.constant(self.xc_dec_init, tf.int32)

        xpi_init = tf.constant(self.xpi_init, tf.float32)
        xpc_init = tf.constant(self.xpc_init, tf.int32)
        slp_init = tf.constant(self.slp_init, tf.int32)
        xpc_dec_init = tf.constant(self.xpc_dec_init, tf.int32)

        # compute weight initializations
        print("Initialize model weights...", flush=True)
        self._model((xi_init, xc_init, sl_init, xc_dec_init,
                     xpi_init, xpc_init, slp_init, xpc_dec_init), init=True)

        # model specification
        print("Define model connections...", flush=True)
        self._model((self.xi, self.xc, self.sl, self.xc_dec,
                     self.xpi, self.xpc, self.slp, self.xpc_dec), init=False)


        print("Marginal bounds...", flush=True)
        # marginal bounds (images)
        self.lxi, self.lxirec, self.lxipen, self.logqi, self.logpi = self._marginal_bound(self.rxi_i, self.xi, None,
                            self.mu_i, self.sigma_i, dtype='image', mode=None, proj=None, scope='marg_xi')

        self.lxpi, self.lxpirec, self.lxpipen, self.logqpi, self.logppi = self._marginal_bound(self.rxi_pi, self.xpi,
                            None, self.mu_pi, self.sigma_pi, dtype='image', mode=None, proj=None, scope='marg_xpi')


        # marginal bounds (captions)  <--- training time
        self.lxc_tr, self.lxcrec_tr, self.lxcpen_tr, self.logqc_tr, self.logpc_tr = self._marginal_bound(
            self.rxc_c, self.xc, self.sl, self.mu_c, self.sigma_c, dtype='caption',
            mode='train', proj=self.proj_c, scope='marg_xc_train')

        self.lxpc_tr, self.lxpcrec_tr, self.lxpcpen_tr, self.logqpc_tr, self.logppc_tr = self._marginal_bound(
            self.rxc_pc, self.xpc, self.slp, self.mu_pc, self.sigma_pc, dtype='caption',
            mode='train', proj=self.proj_pc, scope='marg_xpc_train')


        # marginal bounds (captions)  <--- evaluation time
        self.lxc_te, self.lxcrec_te, self.lxcpen_te, self.logqc_te, self.logpc_te = self._marginal_bound(
            self.rxc_c, self.xc, self.sl, self.mu_c, self.sigma_c, dtype='caption',
            mode='test', proj=self.proj_c, scope='marg_xc_test')

        self.lxpc_te, self.lxpcrec_te, self.lxpcpen_te, self.logqpc_te, self.logppc_te = self._marginal_bound(
            self.rxc_pc, self.xpc, self.slp, self.mu_pc, self.sigma_pc, dtype='caption',
            mode='test', proj=self.proj_pc, scope='marg_xpc_test')


        print("Joint bound...", flush=True)

        # joint bound <--- training time
        self.lxj_tr, self.lxjreci_tr, self.lxjrecc_tr, self.lxjpen_tr, self.logqj_tr, self.logpj_tr = self._joint_bound(
            self.rxi_j, self.xpi, self.rxc_j, self.xpc, self.slp, self.mu_j, self.sigma_j,
            mode='train', proj=self.proj_j, scope='joint_bound')

        # joint bound <--- evaluation time
        self.lxj_te, self.lxjreci_te, self.lxjrecc_te, self.lxjpen_te, self.logqj_te, self.logpj_te = self._joint_bound(
            self.rxi_j, self.xpi, self.rxc_j, self.xpc, self.slp, self.mu_j, self.sigma_j,
            mode='test', proj=self.proj_j, scope='joint_bound')


        print("Translation bounds...", flush=True)

        # translation bound (images)
        self.txi = self._translation_bound(self.rxi_pc, self.xpi, None, dtype='image',
                                           mode=None, proj=None, scope='translate_to_xi')

        # translation bounds (captions) <--- training time
        self.txc_tr = self._translation_bound(self.rxc_pi, self.xpc, self.slp, dtype='caption',
                                           mode='train', proj=self.proj_pi, scope='translate_to_xc')

        # translation bounds (captions) <--- evaluation time
        self.txc_te = self._translation_bound(self.rxc_pi, self.xpc, self.slp, dtype='caption',
                                           mode='test', proj=self.proj_pi, scope='translate_to_xc')


        # loss function
        print("Loss function...", flush=True)
        self.loss = self._loss(scope='loss')

        # optimizer
        print("Optimizer...", flush=True)
        self.step = self._optimizer(self.loss, scope='optimizer')

        # summary variables
        print("Summary variables...", flush=True)
        self.summary_train, self.summary_test = self._summaries()



    def _model(self, xs, init):

        with tf.variable_scope('multimodal_vae') as scope:
            if not init:
                scope.reuse_variables()

            xi, xc, sl, xc_dec, xpi, xpc, slp, xpc_dec = xs

            # encoders
            print("Encoder (caption)...", flush=True)
            mu_pc, sigma_pc, hepc, embed = self._encoder_c(xpc, slp, init=init, scope='xc_enc')
            print("Encoder (image)...", flush=True)
            mu_pi, sigma_pi, hepi = self._encoder_i(xpi, init=init, scope='xi_enc')

            # joint encoder
            print("Joint Encoder...", flush=True)
            mu_j, sigma_j = self._joint_encoder(xhi=hepi, xhc=hepc, init=init, scope='xixc_enc')

            # sample network
            print("Sampler...", flush=True)
            zj = self._sample(mu_j, sigma_j, scope='sample')

            # decoders
            print("Decoders...", flush=True)
            rxi_j, rxi_j_probs = self._decoder_i(zj, init=init, scope='xi_dec')
            rxc_j, rxc_j_probs, proj_j = self._decoder_c(zj, xpc_dec, embed, init=init, scope='xc_dec')


            if not init:

                # unpaired encodings
                print("Encoders (2)...", flush=True)
                mu_i, sigma_i, _ = self._encoder_i(xi, init=init, scope='xi_enc')
                mu_c, sigma_c, _, _ = self._encoder_c(xc, sl, init=init, scope='xc_enc')

                # additional samples
                print("Samplers (2)...", flush=True)
                zi = self._sample(mu_i, sigma_i, scope='sample')
                zc = self._sample(mu_c, sigma_c, scope='sample')
                zpi = self._sample(mu_pi, sigma_pi, scope='sample')
                zpc = self._sample(mu_pc, sigma_pc, scope='sample')

                # reconstructions
                print("Decoders (2)...", flush=True)
                rxi_i, rxi_i_probs = self._decoder_i(zi, init=init, scope='xi_dec')
                rxc_c, rxc_c_probs, proj_c = self._decoder_c(zc, xc_dec, embed, init=init, scope='xc_dec')

                # translations
                rxi_c, rxi_c_probs = self._decoder_i(zc, init=init, scope='xi_dec')
                rxc_i, rxc_i_probs, proj_i = self._decoder_c(zi, xc_dec, embed, init=init, scope='xc_dec')

                # reconstructions (from paired input)
                rxi_pi, rxi_pi_probs = self._decoder_i(zpi, init=init, scope='xi_dec')
                rxc_pc, rxc_pc_probs, proj_pc = self._decoder_c(zpc, xpc_dec, embed, init=init, scope='xc_dec')

                # translations (from paired input)
                rxi_pc, rxi_pc_probs = self._decoder_i(zpc, init=init, scope='xi_dec')
                rxc_pi, rxc_pi_probs, proj_pi = self._decoder_c(zpi, xpc_dec, embed, init=init, scope='xc_dec')

                # final tensors
                print("Assignments...", flush=True)
                self.mu_j, self.sigma_j = (mu_j, sigma_j)
                self.mu_i, self.sigma_i = (mu_i, sigma_i)
                self.mu_c, self.sigma_c = (mu_c, sigma_c)
                self.mu_pi, self.sigma_pi = (mu_pi, sigma_pi)
                self.mu_pc, self.sigma_pc = (mu_pc, sigma_pc)

                self.proj_j, self.proj_c, self.proj_i, self.proj_pc, self.proj_pi = \
                    proj_j, proj_c, proj_i, proj_pc, proj_pi

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

        mu, sigma, he = nw.convolution_coco(x, self.nch, self.n_fmaps_image,
                                            self.n_units_image, self.n_z, init, scope)

        return mu, sigma, he


    def _encoder_c(self, x, sl, init, scope):

        mu, sigma, he, embed_matrix = nw.seq_encoder(x, sl, self.vocab_size, self.embed_size, self.n_units_enc_capt,
                                       self.n_z, self.gru_layers, init, scope)

        return mu, sigma, he, embed_matrix



    def _sample(self, mu0, sigma0, scope):

        with tf.variable_scope(scope):
            n_samples = tf.shape(mu0)[0]
            epsilon = tf.random_normal((n_samples, self.n_z))

            return mu0 + tf.multiply(sigma0, epsilon)


    def _decoder_i(self, z, init, scope):

        nch = self.nch * 256

        z = nw.deconvolution_coco(z, nch, self.n_fmaps_image, self.n_units_image, init, scope)

        logits = tf.reshape(z, shape=[-1, self.nxi, 256])
        parms = tf.nn.softmax(logits, dim=-1)

        return logits, parms


    def _decoder_c(self, z, x_dec, embed, init, scope):

        logits = nw.seq_decoder_cnn(z, x_dec, embed, self.n_fmaps_capt, self.n_units_image, init, scope)

        # logits: batch_size x max_seq_len x n_units

        with tf.variable_scope('projections'):
            if init:
                w = tf.get_variable("w", shape=[self.vocab_size, self.n_fmaps_capt],
                                    initializer=tf.random_normal_initializer(0, 0.05))
                b = tf.get_variable("b", shape=[self.vocab_size], initializer=tf.random_normal_initializer(0, 0.05))
                w = w.initialized_value()
                b = b.initialized_value()

            else:
                w = tf.get_variable("w", shape=[self.vocab_size, self.n_fmaps_capt])
                b = tf.get_variable("b", shape=[self.vocab_size])

            output_projections = (w, b)

        outputs = tf.unstack(logits, axis=1)
        outputs = [tf.matmul(out, tf.transpose(w)) + b  for out in outputs]
        proj_logits = tf.stack(outputs, axis=1)
        # projected logits: batch_size x max_seq_len x vocab_size

        probs = tf.nn.softmax(proj_logits, dim=-1)

        return logits, probs, output_projections


    def _joint_encoder(self, xhi, xhc, init, scope):

        with tf.variable_scope(scope):

            z_mu, z_sigma = nw.joint_coco_encode(xhi, xhc, self.n_units_image, self.n_z, init, scope=scope)

            return z_mu, z_sigma


    def _reconstruction(self, logits, labels, slens, dtype, mode, proj, scope):

        with tf.variable_scope(scope):

            if dtype == 'image':

                labels = tf.cast(labels * 255, dtype=tf.int32)  # integers [0,255] inclusive
                l1 = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

                l1 = tf.reduce_sum(l1, axis=1)
                l1 = tf.reduce_mean(l1, axis=0)

                return l1

            elif dtype == 'caption':

                w, b = proj

                if mode == 'test':
                    # logits: batch_size x max_seq_len x n_units
                    outputs = tf.unstack(logits, axis=1)
                    outputs = [tf.matmul(out, tf.transpose(w)) + b for out in outputs]
                    logits = tf.stack(outputs, axis=1)
                    # projected logits: batch_size x max_seq_len x vocab_size

                    l1 = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

                elif mode == 'train':
                    # logits: batch_size x max_seq_len x n_units
                    # labels: batch_size x max_seq_len

                    batch_size = tf.shape(logits)[0]
                    max_seq_len = logits.get_shape()[1].value
                    n_units = logits.get_shape()[2].value

                    logits = tf.reshape(logits, shape=[batch_size * max_seq_len, n_units])
                    labels = tf.reshape(labels, shape=[-1, 1])

                    l1 = -tf.nn.sampled_softmax_loss(weights=w, biases=b, inputs=logits, labels=labels,
                                                     num_sampled=self.softmax_samples, num_classes=self.vocab_size,
                                                     partition_strategy="div")

                    # l1: (batch_size x max_seq_len) x 1
                    l1 = tf.reshape(l1, shape=[batch_size, max_seq_len])
                    # l1: batch_size x max_seq_len

                else:
                    raise NotImplementedError

                # mask out irrelevant sequence vectors
                max_len = l1.get_shape()[1].value
                mask = tf.sequence_mask(slens, maxlen=max_len, dtype=tf.float32)
                l1 = mask * l1

                # reduce
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


    def _marginal_bound(self, logits, labels, slens, mu, sigma, dtype, mode, proj, scope):

        with tf.variable_scope(scope):

            l1 = self._reconstruction(logits=logits, labels=labels, slens=slens, dtype=dtype,
                                      mode=mode, proj=proj, scope='reconstruction')
            l2, log_q, log_p = self._penalty(mu=mu, sigma=sigma, scope='penalty')
            bound = l1 + l2

            return bound, l1, l2, log_q, log_p


    def _joint_bound(self, xi_logits, xi_labels, xc_logits, xc_labels, slens, mu, sigma, mode, proj, scope):

        with tf.variable_scope(scope):

            l1_xi = self._reconstruction(logits=xi_logits, labels=xi_labels, slens=slens, dtype='image',
                                         mode=None, proj=None, scope='reconstruction_xi')
            l1_xc = self._reconstruction(logits=xc_logits, labels=xc_labels, slens=slens, dtype='caption',
                                         mode=mode, proj=proj, scope='reconstruction_xc')

            l2, log_q, log_p = self._penalty(mu=mu, sigma=sigma, scope='penalty')

            bound = l1_xi + l1_xc + l2

            return bound, l1_xi, l1_xc, l2, log_q, log_p


    def _translation_bound(self, logits, labels, slens, dtype, mode, proj, scope):

        with tf.variable_scope(scope):
            return self._reconstruction(logits=logits, labels=labels, slens=slens, dtype=dtype,
                                        mode=mode, proj=proj, scope='reconstruction')


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
            lxcpen = self._freebits(self.lxcpen_tr, self.logqc_tr, self.logpc_tr, self.alpha)
            lxc = self.lxcrec_tr + lxcpen

            if self.objective == "joint":
                # joint xi and xc
                lxjpen = self._freebits(self.lxjpen_tr, self.logqj_tr, self.logpj_tr, self.alpha)
                lxj = self.lxjreci_tr + self.lxjrecc_tr + lxjpen

                bound = (self.joint_anneal * lxj) + lxi + lxc

            elif self.objective == "translate":
                # marginal x1p
                lxpipen = self._freebits(self.lxpipen,  self.logqpi, self.logppi, self.alpha)
                lxpi = self.lxpirec + lxpipen

                # marginal x2p
                lxpcpen = self._freebits(self.lxpcpen_tr, self.logqpc_tr, self.logppc_tr, self.alpha)
                lxpc = self.lxpcrec_tr + lxpcpen

                bound = (self.txi + lxpc) + (self.txc_tr + lxpi) + lxi + lxc

            else:
                raise NotImplementedError

            loss = -bound

            loss = tf.Print(loss, [loss])   ############################  remove later

            return loss


    def _summaries(self,):
        with tf.variable_scope("summaries"):

            training = []
            evaluation = []

            training.append( tf.summary.scalar('loss', self.loss) )

            if self.objective == "joint":
                training.append( tf.summary.scalar('lower_bound_on_log_p_xi_xc_tr', self.lxj_tr) )
                evaluation.append( tf.summary.scalar('lower_bound_on_log_p_xi_xc_eval', self.lxj_te) )

                training.append( tf.summary.scalar('joint_penalty_tr', self.lxjpen_tr) )
                evaluation.append(tf.summary.scalar('joint_penalty_eval', self.lxjpen_tr))

                training.append(tf.summary.scalar('joint_train', self.lxj_tr))
                evaluation.append(tf.summary.scalar('joint_eval', self.lxj_te))


            elif self.objective == "translate":
                training.append( tf.summary.scalar('lower_bound_on_log_p_xi_xc_ti_train', self.txi + self.lxpc_tr) )
                training.append( tf.summary.scalar('lower_bound_on_log_p_xi_xc_tc_train', self.txc_tr + self.lxpi) )

                evaluation.append( tf.summary.scalar('lower_bound_on_log_p_xi_xc_ti_eval', self.txi + self.lxpc_te) )
                evaluation.append( tf.summary.scalar('lower_bound_on_log_p_xi_xc_tc_eval', self.txc_te + self.lxpi) )

                training.append(tf.summary.scalar('trans_to_xi_train', self.txi))
                training.append(tf.summary.scalar('trans_to_xc_train', self.txc_tr))

                evaluation.append(tf.summary.scalar('trans_to_xi_eval', self.txi))
                evaluation.append(tf.summary.scalar('trans_to_xc_eval', self.txc_te))

            else:
                raise NotImplementedError

            training.append( tf.summary.scalar('marg_xi_train', self.lxi) )
            training.append( tf.summary.scalar('marg_xc_train', self.lxc_tr) )

            training.append( tf.summary.scalar('marg_xpi_train', self.lxpi) )
            training.append( tf.summary.scalar('marg_xpc_train', self.lxpc_tr) )

            evaluation.append(tf.summary.scalar('marg_xpi_eval', self.lxpi))
            evaluation.append(tf.summary.scalar('marg_xpc_eval', self.lxpc_te))

            training = tf.summary.merge(training)
            evaluation = tf.summary.merge(evaluation)

            return training, evaluation


    def _track(self, terms, prefix):

        if self.tracker is not None:

            for name, term in terms.items():
                self.tracker.add(i=self.n_steps, value=term, series_name=prefix+name, run_name=self.name)


    def train(self, xs):
        """
        Performs single training step.
        """
        xi, xc, sl, xc_dec, xi_pairs, xc_pairs, sl_pairs, xc_pairs_dec = xs

        feed = {self.xi: xi, self.xc: xc, self.sl: sl, self.xc_dec: xc_dec,
                self.xpi: xi_pairs, self.xpc: xc_pairs, self.slp: sl_pairs, self.xpc_dec: xc_pairs_dec}
        summary, _ = self.sess.run([self.summary_train, self.step], feed_dict=feed)

        self.tr_writer.add_summary(summary, self.n_steps)

        self.n_steps = self.n_steps + 1


    def test(self, xs):
        """
        Computes lower bound on test data.
        """
        xi, xc, sl, xc_dec = xs

        feed = {self.xi: xi, self.xc: xc, self.sl: sl, self.xc_dec: xc_dec,
                self.xpi: xi, self.xpc: xc, self.slp: sl, self.xpc_dec: xc_dec}
        summary = self.sess.run(self.summary_test, feed_dict=feed)

        self.te_writer.add_summary(summary, self.n_steps)


    def reconstruct(self, xs, mean=False):
        """
        Reconstruct input.
        If one input is None, then this can be interpretted as a translation from the other input.

        xs: (xi, xc ,sl) or (xi, None, None) or (None, xc, sl)
        """
        z = self.encode(xs, mean=mean)
        xi, xc = self.decode(z)

        return xi, xc


    def encode(self, xs, mean=False):
        """
        Encodes xi or xc or both xi and xc to the latent space.
        """
        xi, xc, sl, xc_dec = xs
        feed = {self.xpi: xi, self.xpc: xc, self.slp: sl, self.xpc_dec: xc_dec}
        outputs = self.mu_j, self.zj

        if xi is None and xc is None:
            raise ValueError

        elif xi is None:
            feed = {self.xpc: xc, self.slp: sl, self.xpc_dec: xc_dec}
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

            #feed = {self.zj: z, self.xpc: xc, self.xpc_dec: xc_dec}   #############################
            feed = {self.zj: z, self.xpc_dec: xc}
            pb = self.sess.run(self.rxc_j_probs, feed_dict=feed)   # batch_size x max_seq_len x vocab_size

            pb = pb[:,i,:]
            pb = np.expand_dims(pb, axis=1)

            pb = self._categorical_sampling(pb, n_cats=self.vocab_size)
            pb = np.squeeze(pb)

            xc[:,i] = pb

        return xc
