import tensorflow as tf
import numpy as np

from models import base
from models import networks as nw


class VAE(base.Model):
    """
    Variational Auto-Encoder with 2 inputs

    Arguments:
    n_x, n_z: dimensionality of input and latent variables
    learning_rate: optimizer learning_rate
    n_enc_units: number of hidden units in encoder fully-connected layers
    n_dec_units: number of hidden units in decoder fully-connected layers
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
        n_x = self.args['n_x']
        n_z = self.args['n_z']

        # input placeholders
        self.x = tf.placeholder(tf.float32, [None, n_x], name='x')

        # encoder
        self.z_mu, self.z_var = self._encoder(self.x, n_x, n_z, scope='x_enc', reuse=False)

        # samples
        self.z = self._sample(self.z_mu, self.z_var, n_z, scope='sampler', reuse=False)

        # decoders
        self.rx, self.rx_probs = self._decoder(self.z, n_z, n_x, scope='x_dec', reuse=False)

        # choice of variational bounds
        self.l_x = self._marginal_bound(self.rx, self.x, self.z_mu, self.z_var, scope='marginal_x')

        # training and test bounds
        self.bound = self._training_bound()
        self.test_bound = self._test_bound()

        # loss function
        self.loss = -self.bound

        # optimizer
        self.step = self._optimizer(self.loss)


    def _encoder(self, x, n_x, n_z, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            n_units = self.args['n_enc_units']

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
            n_units = self.args['n_dec_units']

            a1 = self._linear(z, n_z, n_units, "layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)

            a2 = self._linear(h1, n_units, n_units, "layer_2", reuse=reuse)
            h2 = tf.nn.relu(a2)

            logits = self._linear(h2, n_units, n_x, "layer_3", reuse=reuse)
            probs = tf.nn.sigmoid(logits)

            return logits, probs


    def _training_bound(self,):
        return tf.reduce_mean(self.l_x, axis=0)


    def _test_bound(self,):
        return tf.reduce_mean(self.l_x, axis=0)


    def _marginal_bound(self, logits, labels, mean, var, scope='marginal_bound', reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            l1 = self._reconstruction_loss(logits=logits, labels=labels)

            l2 = 0.5 * tf.reduce_sum(1 + tf.log(var) - tf.square(mean) - var, axis=1)

            return l1 + l2


    def _reconstruction_loss(self, logits, labels):

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


    def _summaries(self,):

        with tf.variable_scope("summaries", reuse=False):
            tf.summary.scalar('training_curve', self.bound)
            tf.summary.scalar('lower_bound_on_log_p_x_y', self.test_bound)

            return tf.summary.merge_all()


    def _linear(self, x, n_x, n_w, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            w = tf.get_variable("W", shape=[n_x,n_w],
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            b = tf.get_variable("b", shape=[n_w], initializer=tf.constant_initializer(0.1))

            return tf.matmul(x, w) + b


    def train(self, x, write=True):
        """
        Performs single training step.
        """
        feed = {self.x: x}
        outputs = [self.summary, self.step, self.bound, self.test_bound]

        summary, _, curve, bound = self.sess.run(outputs, feed_dict=feed)
        if write:
            self.tr_writer.add_summary(summary, self.n_steps)
        self.n_steps = self.n_steps + 1

        return curve, bound


    def test(self, x):
        """
        Computes lower bound on test data.
        """
        feed = {self.x: x}
        outputs = [self.summary, self.test_bound]

        summary, test_bound = self.sess.run(outputs, feed_dict=feed)
        self.te_writer.add_summary(summary, self.n_steps)

        return test_bound


    def reconstruct(self, x):
        """
        Reconstruct x.
        """
        feed = {self.x: x}
        return self.sess.run(self.rx_probs, feed_dict=feed)


    def encode(self, x):
        """
        Encode x1.
        """
        feed = {self.x: x}
        return self.sess.run(self.z_mu, feed_dict=feed)


    def decode(self, z):
        """
        Decodes x1 and x2.
        """
        feed = {self.z: z}
        return self.sess.run(self.rx_probs, feed_dict=feed)




class VAECNN(VAE):

    """
    Variational Auto-Encoder with PixelCNN Decoder. Bernouli Output distribution.

    Arguments:
    n_x1, n_x2, n_z: dimensionality of input and latent variables
    learning_rate: optimizer learning_rate
    n_enc_units: number of hidden units in encoder fully-connected layers
    n_dec_units: number of hidden units in decoder fully-connected layers
    image_dim: shape of input image
    filter_w: width of convolutional filter (with width = height)
    n_dec_layers: number of layers in decoder
    """
    def __init__(self, arguments, name="VAECNN", session=None, log_dir=None, model_dir=None):

        super(VAECNN, self).__init__(arguments=arguments, name=name, session=session,
                                              log_dir=log_dir, model_dir=model_dir)


    def _encoder(self, x, n_x, n_z, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            n_units = self.args['n_enc_units']
            im_dim = self.args['image_dim']
            in_ch = im_dim[2]

            x_image = tf.reshape(x, shape=[-1]+im_dim)

            h1 = nw.conv_pool(x_image, in_ch, 32, 'layer_1', reuse=reuse)

            h2 = nw.conv_pool(h1, 32, 32, 'layer_2', reuse=reuse)

            dim = h2.get_shape()[1].value * h2.get_shape()[2].value * h2.get_shape()[3].value
            flat = tf.reshape(h2, shape=[-1,dim])

            fc = self._linear(flat, dim, n_units, 'layer_3', reuse=reuse)
            h3 = tf.nn.relu(fc)

            mean = self._linear(h3, n_units, n_z, "mean_layer", reuse=reuse)

            a3 = self._linear(h3, n_units, n_z, "var_layer", reuse=reuse)
            var = tf.nn.softplus(a3)

            return mean, var


    def _decoder(self, z, n_z, n_x, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):

            n_units = self.args['n_dec_units']
            im_dim = self.args['image_dim']
            n_layers = self.args['n_dec_layers']
            k = self.args['filter_w']
            out_ch = im_dim[2]

            a1 = self._linear(z, n_z, n_units, "layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)

            z_units = im_dim[0] * im_dim[1] * im_dim[2]
            a2 = self._linear(h1, n_units, z_units, "layer_2", reuse=reuse)
            h2 = tf.nn.relu(a2)

            z_image = tf.reshape(h2, shape=[-1]+im_dim)

            logits, probs = self._decoder_cnn(z_image, n_layers, k, out_ch, 'pixel_cnn', reuse)

            return logits, probs


    def _decoder_cnn(self, x, n_layers, k, out_ch, scope, reuse):
        """
        Combines CNN network with output distribution.

        x: tensor of images
        n_layers: number of layers in CNN
        k: convolution filter size
        out_ch: network output channels
        """
        c = nw.pixel_cnn(x, n_layers, k, out_ch, scope, reuse=reuse)

        n_c = c.get_shape()[1].value * c.get_shape()[2].value * c.get_shape()[3].value

        logits = tf.reshape(c, shape=[-1, n_c])
        probs = tf.nn.sigmoid(logits)

        return logits, probs




class VAECNN_Color(VAECNN):

    """
    Variational Auto-Encoder with PixelCNN Decoder. 256-way Categorical output distribution.

    Arguments:
    n_x1, n_x2, n_z: dimensionality of input and latent variables
    learning_rate: optimizer learning_rate
    n_enc_units: number of hidden units in encoder fully-connected layers
    n_dec_units: number of hidden units in decoder fully-connected layers
    image_dim: shape of input image
    filter_w: width of convolutional filter (with width = height)
    n_dec_layers: number of layers in decoder
    """
    def __init__(self, arguments, name="VAECNN_Color", session=None, log_dir=None, model_dir=None):

        super(VAECNN_Color, self).__init__(arguments=arguments, name=name, session=session,
                                              log_dir=log_dir, model_dir=model_dir)


    def _decoder_cnn(self, x, n_layers, k, out_ch, scope, reuse):
        """
        Combines CNN network with output distribution.
        """
        logits = nw.pixel_cnn_categorical(x, n_layers, k, out_ch, n_cats=256, scope=scope, reuse=reuse)
        probs = tf.nn.softmax(logits, dim=-1)

        return logits, probs


    def _reconstruction_loss(self, logits, labels):

        # note: labels have dimension [batch_size, h*w*ch]

        # scale and discretize pixel intensities
        labels = tf.cast(labels * 255, dtype=tf.int32)

        # reshape to images
        labels = tf.reshape(labels, shape=[-1]+self.args['image_dim'])

        return tf.reduce_sum(-tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels), axis=[1,2,3])


