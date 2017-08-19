import tensorflow as tf
import numpy as np

from models import vae
from models import layers as nw


class JointVAE(vae.VAE):
    """
    Variational Auto-Encoder with Two Input Modalities
    """
    def __init__(self, arguments, name, tracker, init_minibatches, session=None, log_dir=None, model_dir=None):

        self.x1_init, self.x2_init, self.x1p_init, self.x2p_init = init_minibatches

        # constructor
        super(JointVAE, self).__init__(arguments=arguments, name=name, tracker=tracker, init_minibatch=None,
                                       session=session, log_dir=log_dir, model_dir=model_dir)


    def _initialize(self,):
        # additional parameters
        self.objective = self.args["objective"]

        # input placeholders
        self.x1 = tf.placeholder(tf.float32, [None, self.n_x], name='x1')
        self.x2 = tf.placeholder(tf.float32, [None, self.n_x], name='x2')
        self.x1p = tf.placeholder(tf.float32, [None, self.n_x], name='x1_paired')
        self.x2p = tf.placeholder(tf.float32, [None, self.n_x], name='x2_paired')

        # data-dependent weight initialization (Salisman, Kingma - 2016)
        x1_init = tf.constant(self.x1_init, tf.float32)
        x2_init = tf.constant(self.x2_init, tf.float32)
        x1p_init = tf.constant(self.x1p_init, tf.float32)
        x2p_init = tf.constant(self.x2p_init, tf.float32)

        # compute weight initializations
        self._model((x1_init, x2_init, x1p_init, x2p_init), init=True)

        # model specification
        self._model((self.x1, self.x2, self.x1p, self.x2p), init=False)


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
            self.rx1_12, self.x1p, self.rx2_12, self.x2p,
            self.z12_mu, self.z12_sigma, self.log_q12, self.z12, scope='joint_bound')

        # translation bounds

        self.tx1 = self._translation_bound(self.rx1_2p, self.x1p, scope='translate_to_x1')
        self.tx2 = self._translation_bound(self.rx2_1p, self.x2p, scope='translate_to_x2')

        # loss function
        self.loss = self._loss(scope='loss')

        # optimizer
        self.step = self._optimizer(self.loss)

        # summary variables
        self.summary = self._summaries()


    def _model(self, xs, init):

        with tf.variable_scope('joint_autoencoder') as scope:
            if not init:
                scope.reuse_variables()

            extra = self.args['flow']

            x1, x2, x1p, x2p = xs

            # encoders
            z1p_mu, z1p_sigma, h1p, he1p = self._encoder(x1p, init=init, scope='x1_enc')
            z2p_mu, z2p_sigma, h2p, he2p = self._encoder(x2p, init=init, scope='x2_enc')

            # joint encoder
            z12_mu, z12_sigma, h12 = self._joint_encoder(z1p_mu, z1p_sigma, h1p, z2p_mu, z2p_sigma, h2p,
                                x1, x2, x1h=he1p, x2h=he2p,
                                extra=extra, init=init, scope='x1x2_enc')

            # sample network
            z12, log_q12 = self._sample(z12_mu, z12_sigma, h12, init=init, scope='sample')

            # decoders
            rx1_12, rx1_12_probs = self._decoder(z12, x1p, init=init, scope='x1_dec')
            rx2_12, rx2_12_probs = self._decoder(z12, x2p, init=init, scope='x2_dec')

        if not init:

            with tf.variable_scope('joint_autoencoder') as scope:

                scope.reuse_variables()

                # unpaired encodings
                z1_mu, z1_sigma, h1, _ = self._encoder(x1, init=init, scope='x1_enc')
                z2_mu, z2_sigma, h2, _ = self._encoder(x2, init=init, scope='x2_enc')

                # additional samples
                z1, log_q1 = self._sample(z1_mu, z1_sigma, h1, init=init, scope='sample')
                z2, log_q2 = self._sample(z2_mu, z2_sigma, h2, init=init, scope='sample')
                z1p, log_q1p = self._sample(z1p_mu, z1p_sigma, h1p, init=init, scope='sample')
                z2p, log_q2p = self._sample(z2p_mu, z2p_sigma, h2p, init=init, scope='sample')

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



    def _joint_encoder(self, mu1, sigma1, h1, mu2, sigma2, h2, x1, x2, x1h, x2h, extra, init, scope):

        with tf.variable_scope(scope):

            jtype = self.args['joint_type']

            n_units = self.args['n_units']
            n_z = self.n_z

            if jtype == 'constrain':
                z_mu, z_sigma = self._constrain(mu1, sigma1, mu2, sigma2, scope='constrain')

                h12 = None
                if h1 is not None and h2 is not None:
                    h12 = tf.nn.elu(h1 + h2)

            elif jtype == 'small':
                z_mu, z_sigma, h12 = nw.joint_fc_encode(x1h, x2h, n_units, n_z, extra, init, scope='small_enc')

            elif jtype == 'large':
                if self.dataset == 'halved_mnist':
                    raise NotImplementedError

                elif self.dataset == 'mnist':
                    raise NotImplementedError

                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

            return z_mu, z_sigma, h12



    def _constrain(self, x1_mu, x1_sigma, x2_mu, x2_sigma, scope):
        """
        Computes mean and standard deviation of the product of two Gaussians.
        """
        with tf.variable_scope(scope):
            x1_var = tf.square(x1_sigma)
            x2_var = tf.square(x2_sigma)
            x1v_inv = tf.reciprocal(x1_var)
            x2v_inv = tf.reciprocal(x2_var)
            x12_var = tf.reciprocal(x1v_inv + x2v_inv)
            xx = tf.multiply(x1v_inv, x1_mu)
            yy = tf.multiply(x2v_inv, x2_mu)
            x12_mu = tf.multiply(x12_var, xx + yy)
            x12_sigma = tf.sqrt(x12_var)

            return x12_mu, x12_sigma


    def _marginal_bound(self, logits, labels, mu, sigma, log_q, z_K, scope):

        with tf.variable_scope(scope):

            l1 = self._reconstruction(logits=logits, labels=labels, scope='reconstruction')
            l2, log_q, log_p = self._penalty(mu=mu, sigma=sigma, log_q=log_q, z_K=z_K, scope='penalty')
            bound = l1 + l2

            return bound, l1, l2, log_q, log_p


    def _joint_bound(self, x1_logits, x1_labels, x2_logits, x2_labels, mu, sigma, log_q, z_K, scope):

        with tf.variable_scope(scope):

            l1_x1 = self._reconstruction(logits=x1_logits, labels=x1_labels, scope='reconstruction_x1')
            l1_x2 = self._reconstruction(logits=x2_logits, labels=x2_labels, scope='reconstruction_x2')

            l2, log_q, log_p = self._penalty(mu=mu, sigma=sigma, log_q=log_q, z_K=z_K, scope='penalty')

            bound = l1_x1 + l1_x2 + l2

            return bound, l1_x1, l1_x2, l2, log_q, log_p


    def _translation_bound(self, logits, labels, scope):

        with tf.variable_scope(scope):
            return self._reconstruction(logits=logits, labels=labels, scope='reconstruction')


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
            alpha = self.args['anneal']

            temp_weight = self.args['temp_weight']

            # marginal x1
            lx1pen = self._freebits(self.lx1pen, self.logq1, self.logp1, alpha)
            lx1 = self.lx1rec + lx1pen

            # marginal x2
            lx2pen = self._freebits(self.lx2pen, self.logq2, self.logp2, alpha)
            lx2 = self.lx2rec + lx2pen

            if self.objective == "joint":
                # joint x1 and x2
                lx12pen = self._freebits(self.lx12pen, self.logq12, self.logp12, alpha)
                lx12 = self.lx12rec1 + self.lx12rec2 + lx12pen

                bound = lx12 + lx1 + lx2

            elif self.objective == "translate":
                # marginal x1p
                lx1ppen = self._freebits(self.lx1ppen,  self.logq1p, self.logp1p, alpha)
                lx1p = self.lx1prec + lx1ppen

                # marginal x2p
                lx2ppen = self._freebits(self.lx2ppen, self.logq2p, self.logp2p, alpha)
                lx2p = self.lx2prec + lx2ppen

                bound = (self.tx1 + lx2p) + (self.tx2 + lx1p) + lx1 + lx2

            else:
                raise NotImplementedError

            loss = -bound
            return loss


    def _summaries(self,):

        with tf.variable_scope("summaries"):
            tf.summary.scalar('loss_(ignore_test)', self.loss)

            if self.objective == "joint":
                tf.summary.scalar('lower_bound_on_log_p_x_y', self.lx12)

            elif self.objective == "translate":
                tf.summary.scalar('lower_bound_on_log_p_x_y_t1', self.tx1 + self.lx2p)
                tf.summary.scalar('lower_bound_on_log_p_x_y_t2', self.tx2 + self.lx1p)

            else:
                raise NotImplementedError

            tf.summary.scalar('marg_x1_(ignore_test)', self.lx1)
            tf.summary.scalar('marg_x2_(ignore_test)', self.lx2)
            tf.summary.scalar('marg_x1p', self.lx1p)
            tf.summary.scalar('marg_x2p', self.lx2p)

            tf.summary.scalar('joint', self.lx12)
            tf.summary.scalar('trans_to_x1', self.tx1)
            tf.summary.scalar('trans_to_x2', self.tx2)

            return tf.summary.merge_all()


    def train(self, xs):
        """
        Performs single training step.
        """
        x1, x2, x1_pairs, x2_pairs = xs

        feed = {self.x1: x1, self.x2: x2, self.x1p: x1_pairs, self.x2p: x2_pairs}
        outputs = [self.summary, self.step, self.loss, self.lx1, self.lx2, self.lx12, self.tx1, self.tx2, self.lx1p, self.lx2p]

        summary, _, loss, lx1, lx2, lx12, tx1, tx2, lx1p, lx2p = self.sess.run(outputs, feed_dict=feed)

        if self.objective == 'joint':
            bound = lx12
            terms = {'lower_bound_on_log_p_x_y': bound, 'loss': loss,
                     'lx1': lx1, 'lx2': lx2, 'lx12': lx12, 'tx1': tx1, 'tx2': tx2}
        else:  # translate
            bound_t1 = tx1 + lx2p
            bound_t2 = tx2 + lx1p
            terms = {'lower_bound_on_log_p_x_y_t1': bound_t1, 'lower_bound_on_log_p_x_y_t2': bound_t2, 'loss': loss,
                     'lx1': lx1, 'lx2': lx2, 'lx12': lx12, 'tx1': tx1, 'tx2': tx2}

        self._track(terms, prefix='train_')
        self.tr_writer.add_summary(summary, self.n_steps)

        self.n_steps = self.n_steps + 1


    def test(self, xs):
        """
        Computes lower bound on test data.
        """
        x1, x2 = xs

        feed = {self.x1: x1, self.x2: x2, self.x1p: x1, self.x2p: x2}
        outputs = [self.summary, self.loss, self.lx1p, self.lx2p, self.lx12, self.tx1, self.tx2, self.lx1p, self.lx2p]

        summary, loss, lx1, lx2, lx12, tx1, tx2, lx1p, lx2p = self.sess.run(outputs, feed_dict=feed)

        if self.objective == 'joint':
            bound = lx12
            terms = {'lower_bound_on_log_p_x_y': bound, 'loss': loss,
                     'lx1': lx1, 'lx2': lx2, 'lx12': lx12, 'tx1': tx1, 'tx2': tx2}
        else:  # translate
            bound_t1 = tx1 + lx2p
            bound_t2 = tx2 + lx1p
            terms = {'lower_bound_on_log_p_x_y_t1': bound_t1, 'lower_bound_on_log_p_x_y_t2': bound_t2, 'loss': loss,
                     'lx1': lx1, 'lx2': lx2, 'lx12': lx12, 'tx1': tx1, 'tx2': tx2}

        self._track(terms, prefix='test_')
        self.te_writer.add_summary(summary, self.n_steps)


    def reconstruct(self, xs):
        """
        Reconstruct input.
        If one input is None, then this can be interpretted as a translation from the other input.

        xs: (x1, x2) or (x1, None) or (None, x2)
        """
        z = self.encode(xs, mean=False)
        x1, x2 = self.decode(z)

        return x1, x2


    def encode(self, xs, mean=False):
        """
        Encodes x1 or x2 or both x1 and x2 to the latent space.
        """
        x1, x2 = xs
        feed = {self.x1p: x1, self.x2p: x2}
        outputs = self.z12_mu, self.z12

        if x1 is None and x2 is None:
            raise ValueError

        elif x1 is None:
            feed = {self.x2p: x2}
            outputs = self.z2p_mu, self.z2p

        elif x2 is None:
            feed = {self.x1p: x1}
            outputs = self.z1p_mu, self.z1p

        if mean:
            assert self.is_flow == False
            return self.sess.run(outputs[0], feed_dict=feed)
        else:
            return self.sess.run(outputs[1], feed_dict=feed)


    def decode(self, z):

        if self.is_autoregressive:
            x1 = np.random.rand(z.shape[0], self.n_x)
            x2 = np.random.rand(z.shape[0], self.n_x)

            x1, x2 = self._joint_autoregressive_sampling(z, x1, x2, n_pixels=0,
                    z_var=self.z12, p1_var=self.rx1_12_probs, p2_var=self.rx2_12_probs)

        else:
            feed = {self.z12: z}
            outputs = [self.rx1_12_probs, self.rx2_12_probs]
            rx1, rx2 = self.sess.run(outputs, feed_dict=feed)

            x1 = self._factorized_sampling(rx1)
            x2 = self._factorized_sampling(rx2)

        return x1, x2


    def _joint_autoregressive_sampling(self, z, x1, x2, n_pixels, z_var, p1_var, p2_var):
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

        x1 = x1.copy()
        x2 = x2.copy()
        x1_empty, x2_empty = self._empty_like(x1, x2)

        for i in range(remain):

            feed = {z_var: z, self.x1p: x1, self.x2p: x2, self.x1: x1_empty, self.x2: x2_empty}
            p1, p2 = self.sess.run([p1_var, p2_var], feed_dict=feed)

            hp, wp = _locate_2d(n_pixels + i, w)

            x1 = np.reshape(x1, newshape=[-1, h, w, ch])
            x2 = np.reshape(x2, newshape=[-1, h, w, ch])

            if self.n_ch == 1:
                p1 = np.reshape(p1, newshape=[-1, h, w, ch])
                p1 = p1[:, hp, wp, :]
                x1[:, hp, wp, :] = np.random.binomial(n=1, p=p1)

                p2 = np.reshape(p2, newshape=[-1, h, w, ch])
                p2 = p2[:, hp, wp, :]
                x2[:, hp, wp, :] = np.random.binomial(n=1, p=p2)

            elif self.distribution == 'discrete':
                p1 = np.reshape(p1, newshape=[-1, h, w, ch, 256])
                p1 = p1[:, hp, wp, :, :]
                x1[:, hp, wp, :] = self._categorical_sampling(p1)

                p2 = np.reshape(p2, newshape=[-1, h, w, ch, 256])
                p2 = p2[:, hp, wp, :, :]
                x2[:, hp, wp, :] = self._categorical_sampling(p2)

            elif self.distribution == 'continuous':
                p1 = np.reshape(p1, newshape=[-1, h, w, ch])
                x1[:, hp, wp, :] = p1[:, hp, wp, :]

                p2 = np.reshape(p2, newshape=[-1, h, w, ch])
                x2[:, hp, wp, :] = p2[:, hp, wp, :]

            else:
                raise NotImplementedError

            x1 = np.reshape(x1, newshape=[-1, n_x])
            x2 = np.reshape(x2, newshape=[-1, n_x])

        return x1, x2


    def _empty_like(self, x1, x2):

        x1_shape = list(x1.shape)
        x1_shape[0] = 0
        x2_shape = list(x2.shape)
        x2_shape[0] = 0
        x1_empty = np.zeros(shape=x1_shape)
        x2_empty = np.zeros(shape=x2_shape)

        return x1_empty, x2_empty
