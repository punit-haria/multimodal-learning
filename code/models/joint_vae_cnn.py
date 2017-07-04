import tensorflow as tf
from models import modules as mod
from models.joint_vae import JointVAE


class JointVAE_CNN(JointVAE):
    
    def __init__(self, input_dim, latent_dim, input_2d_dim,
        learning_rate, n_hidden_units=200, joint_strategy='constrain',
        name="JointVAE_CNN", session=None, log_dir=None, model_dir=None):
        """
        Same as the JointVAE with the exception that encoders and decoders 
        now incorporate 2 convolutional layers.

        input_2d_dim: tuple of input image dimensions (height, width, n_channels)
        """
        self._h, self._w, self._nc = input_2d_dim

        super(JointVAE_CNN, self).__init__(input_dim, latent_dim, learning_rate, n_hidden_units,        
            joint_strategy, name, session, log_dir, model_dir)


    def _q_z_x(self, X, x_dim, z_dim, n_hidden, scope, reuse):
        """
        Inference network using convolution.

        X: input data
        x_dim: dimensionality of input space 
        z_dim: dimensionality of latent space
        n_hidden: number of hidden units in each layer
        """
        with tf.variable_scope(scope, reuse=reuse):
            X_image = tf.reshape(X, [-1, self._h, self._w, self._nc])

            a1 = self._conv_pool(X_image, self._nc, 16, scope="layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)

            a2 = self._conv_pool(h1, 16, 16, scope="layer_2", reuse=reuse)
            h2 = tf.nn.relu(a2)

            dim = h2.get_shape()[1].value * h2.get_shape()[2].value * h2.get_shape()[3].value
            flat = tf.reshape(h2, [-1, dim])

            fc = self._affine_map(flat, dim, n_hidden, "fc_layer", reuse=reuse)
            h3 = tf.nn.relu(fc)

            z_mean = self._affine_map(h3, n_hidden, z_dim, "mean_layer", reuse=reuse)

            a3_var = self._affine_map(h3, n_hidden, z_dim, "var_layer", reuse=reuse)
            z_var = tf.nn.softplus(a3_var)

            return z_mean, z_var, h3


    def _p_x_z(self, Z, z_dim, x_dim, n_hidden, scope, reuse):
        """
        Generator network using deconvolution.

        Z: latent data
        z_dim: dimensionality of latent space
        x_dim: dimensionality of output space
        n_hidden: number of hidden units in each layer
        """
        with tf.variable_scope(scope, reuse=reuse):

            b_size = tf.shape(Z)[0]

            a1 = self._affine_map(Z, z_dim, n_hidden, "layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)

            _h = int(self._h / 4) 
            _w = int(self._w / 4)

            a2 = self._affine_map(h1, n_hidden, _h*_w*16, "layer_2", reuse=reuse)

            Z_2d = tf.reshape(a2, [-1, _h, _w, 16])

            r1 = tf.nn.relu(Z_2d)

            up_1 = self._depool(r1, scope="depool_1", reuse=reuse)
            d1 = self._deconv(up_1, 16, 16, scope="deconv_1", reuse=reuse)
            d1.set_shape([None, 14, 14, 16])

            r2 = tf.nn.relu(d1)

            up_2 = self._depool(r2, scope="depool_2", reuse=reuse)
            d2 = self._deconv(up_2, 16, self._nc, scope="deconv_2", reuse=reuse)
            d2.set_shape([None, 28, 28, self._nc])

            x_logits = tf.reshape(d2, [-1, self._h * self._w * self._nc])
            x_probs = tf.nn.sigmoid(x_logits)

            return x_logits, x_probs

    
    def _conv_pool(self, input, in_ch, out_ch, scope, reuse):
        """
        Combined convolution and pooling layer.
        """
        with tf.variable_scope(scope, reuse=reuse):
            C = self._conv(input, in_ch, out_ch, scope="conv", reuse=reuse)
            return self._pool(C, scope="pool", reuse=reuse)


    def _conv(self, input, in_ch, out_ch, scope, reuse):
        """
        Convolution layer

        in_ch/out_ch: number of input and output channels
        """ 
        with tf.variable_scope(scope, reuse=reuse):
            w = self._weight([3, 3, in_ch, out_ch], reuse)
            b = self._bias([out_ch])

            return tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='SAME') + b


    def _pool(self, input, scope, reuse):
        """
        Max pooling layer
        """
        with tf.variable_scope(scope, reuse=reuse):
            return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


    def _deconv(self, input, in_ch, out_ch, scope, reuse):
        """
        Deconvolution layer, i.e. transpose of convolution

        in_ch/out_ch: number of input and output channels
        """
        with tf.variable_scope(scope, reuse=reuse):
            batch_size = tf.shape(input)[0]
            height = input.get_shape()[1].value 
            width = input.get_shape()[2].value 

            W = self._weight([3, 3, out_ch, in_ch], reuse)
            b = self._bias([out_ch])
            
            out_shape = [batch_size, height, width, out_ch]

            return tf.nn.conv2d_transpose(input, W, output_shape=out_shape, 
                strides=[1,1,1,1], padding='SAME') + b


    def _depool(self, input, factor=2, scope="depool", reuse=False):
        """
        Taken from https://gist.github.com/kastnerkyle/f3f67424adda343fef40 

        luke perforated upsample
        http://www.brml.org/uploads/tx_sibibtex/281.pdf
        """
        with tf.variable_scope(scope, reuse=reuse):

            X = tf.transpose(input, perm=[0,3,1,2])

            batch_size = tf.shape(X)[0]
            channels = X.get_shape()[1].value
            height = X.get_shape()[2].value
            width = X.get_shape()[3].value

            stride = height
            offset = width
            in_dim = stride * offset
            out_dim = in_dim * factor * factor

            upsamp_matrix = tf.get_variable(name='upsamp', shape=[in_dim, out_dim], 
                initializer=tf.zeros_initializer(), trainable=False)
            step = factor * factor
            upsamp_matrix = upsamp_matrix[:, ::step].assign(tf.ones(tf.shape(upsamp_matrix[:, ::step])))

            flat = tf.reshape(X, [batch_size, channels, height * width]) 
            up_flat = tf.tensordot(flat, upsamp_matrix, axes=1)

            upsamp = tf.reshape(up_flat, [batch_size, channels,
                                        height * factor, width * factor])

            return tf.transpose(upsamp, perm=[0,2,3,1])


    def _weight(self, shape, reuse):
        """
        Initialize weight variable.
        """
        W = tf.get_variable("W", shape=shape, 
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        if not reuse:
            tf.summary.histogram("weight", W)
            
        return W


    def _bias(self, shape):
        """
        Initialize bias variable. 
        """
        return tf.get_variable("b", shape=shape, initializer=tf.constant_initializer(0.1))