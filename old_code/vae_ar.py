import tensorflow as tf
import numpy as np

from models.vae import  VAE
from models import layers as nw


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






