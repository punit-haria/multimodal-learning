import tensorflow as tf

from models import vae
from models import layers as nw



class VAECNN_Color(vae.VAE_CNN):

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


    def _pixel_cnn(self, x, z, scope, reuse):
        """
        Combines CNN network with output distribution.
        """
        n_layers = self.args['n_pixelcnn_layers']
        k = self.args['filter_w']

        n_cats = 256
        n_channels = self.n_ch * n_cats
        cnn = nw.pixel_cnn(x, n_layers, ka=7, kb=3, out_ch=n_channels, scope=scope, reuse=reuse)

        h = cnn.get_shape()[1].value
        w = cnn.get_shape()[2].value

        logits = tf.reshape(cnn, shape=[-1, h, w, self.n_ch, n_cats])
        probs = tf.nn.softmax(logits, dim=-1)

        return logits, probs


    def _reconstruction(self, logits, labels, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            # note: labels have dimension [batch_size, h*w*ch]

            # scale and discretize pixel intensities
            labels = tf.cast(labels * 255, dtype=tf.int32)

            # reshape to images
            labels = tf.reshape(labels, shape=[-1, self.h, self.w, self.n_ch])

            l1 = tf.reduce_sum(-tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels), axis=[1,2,3])

            return tf.reduce_mean(l1, axis=0)