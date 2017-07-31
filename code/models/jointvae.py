import tensorflow as tf
import numpy as np

from models import vae
from models import layers as nw


class JointVAE(vae.VAE):
    """
    Variational Auto-Encoder with Two Input Modalities
    """
    def __init__(self, arguments, name, tracker, init_minibatches, session=None, log_dir=None, model_dir=None):

        super(JointVAE, self).__init__(arguments=arguments, name=name, tracker=tracker, init_minibatch=None,
                                       session=session, log_dir=log_dir, model_dir=model_dir)

        # initialization minibatches
        self.x1_init, self.x2_init, self.x1p_init, self.x2p_init = init_minibatches

        # additional parameters
        self.objective = self.args["objective"]


    def _initialize(self,):

        # input placeholders
        self.x1 = tf.placeholder(tf.float32, [None, self.n_x], name='x1')
        self.x2 = tf.placeholder(tf.float32, [None, self.n_x], name='x2')
        self.x1p = tf.placeholder(tf.float32, [None, self.n_x], name='x1_paired')
        self.x2p = tf.placeholder(tf.float32, [None, self.n_x], name='x2_paired')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

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

        # variational bound at test time
        if self.objective == "joint":
            self.bound = self.lx12

        elif self.objective == "average":
            self.bound = self.tx1 + self.tx2   # divide by 2?

        else:
            raise NotImplementedError

        # loss function
        self.loss = self._loss(scope='loss')

        self.step = self._optimizer(self.loss)


    def _model(self, xs, init):

        with tf.variable_scope('joint_autoencoder') as scope:
            if not init:
                scope.reuse_variables()

            x1, x2, x1p, x2p = xs

            z1p_mu, z1p_sigma, h1p = self._encoder(x1p, init=init, scope='x1_enc')
            z2p_mu, z2p_sigma, h2p = self._encoder(x2p, init=init, scope='x2_enc')


        with tf.variable_scope('joint_autoencoder') as scope:

            scope.reuse_variables()

            z1_mu, z1_sigma, h1 = self._encoder(x1, init=init, scope='x1_enc')
            z2_mu, z2_sigma, h2 = self._encoder(x2, init=init, scope='x2_enc')


        with tf.variable_scope('joint_autoencoder') as scope:
            if not init:
                scope.reuse_variables()

            z12_mu, z12_sigma = self._constrain(z1p_mu, z1p_sigma, z2p_mu, z2p_sigma, scope='x1x2_enc')
            h12 = tf.nn.elu(h1p + h2p)

            z12, log_q12 = self._sample(z12_mu, z12_sigma, h12, init=init, scope='sample')


        with tf.variable_scope('joint_autoencoder') as scope:

            scope.reuse_variables()

            z1, log_q1 = self._sample(z1_mu, z1_sigma, h1, init=init, scope='sample')
            z2, log_q2 = self._sample(z2_mu, z2_sigma, h2, init=init, scope='sample')
            z1p, log_q1p = self._sample(z1p_mu, z1p_sigma, h1p, init=init, scope='sample')
            z2p, log_q2p = self._sample(z2p_mu, z2p_sigma, h2p, init=init, scope='sample')


        with tf.variable_scope('joint_autoencoder') as scope:
            if not init:
                scope.reuse_variables()

            rx1_12, rx1_12_probs = self._decoder(z12, x1p, init=init, scope='x1_dec')
            rx2_12, rx2_12_probs = self._decoder(z12, x2p, init=init, scope='x2_dec')

        with tf.variable_scope('joint_autoencoder') as scope:

            scope.reuse_variables()

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

        if not init:
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

            elif self.objective == "average":
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
            tf.summary.scalar('lower_bound', self.bound)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('reconstruction', self.l1)
            tf.summary.scalar('penalty', self.l2)

            tf.summary.scalar('sigma0', tf.reduce_mean(self.z_sigma))

            c = 0.5 * np.log(2*np.pi)
            lq = tf.reduce_sum(self.log_q - c, axis=1)
            tf.summary.scalar('penalty_log_q', tf.reduce_mean(lq, axis=0))
            lp = tf.reduce_sum(self.log_p - c, axis=1)
            tf.summary.scalar('penalty_log_p', tf.reduce_mean(lp, axis=0))

            return tf.summary.merge_all()


    def _summaries(self,):

        with tf.variable_scope("summaries", reuse=False):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('lower_bound_on_log_p_x_y', self.bound)

            return tf.summary.merge_all()




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
        x1_shape = list(x1.shape)
        x1_shape[0] = 0
        x2_shape = list(x2.shape)
        x2_shape[0] = 0
        x1_empty = np.zeros(shape=x1_shape)
        x2_empty = np.zeros(shape=x2_shape)

        feed = {self.x1: x1_empty, self.x2: x2_empty, self.x1p: x1, self.x2p: x2}
        outputs = [self.summary, self.test_bound]

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


