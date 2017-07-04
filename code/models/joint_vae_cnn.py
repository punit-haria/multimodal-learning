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

            w1 = self._weight([3, 3, self._nc, 16], "layer_1", reuse=reuse)
            b1 = self._bias([16], "layer_1", reuse=reuse)
            c1 = tf.nn.conv2d(X_image, w1, strides=[1,1,1,1], padding='SAME') + b1
            self.r1 = tf.nn.relu(c1)

            m1, self.m1_argmax = tf.nn.max_pool_with_argmax(self.r1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            w2 = self._weight([3, 3, 16, 16], "layer_2", reuse=reuse)
            b2 = self._bias([16], "layer_2", reuse=reuse)
            c2 = tf.nn.conv2d(m1, w2, strides=[1,1,1,1], padding='SAME') + b2
            self.r2 = tf.nn.relu(c2)

            m2, self.m2_argmax = tf.nn.max_pool_with_argmax(self.r2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            dim = m2.get_shape()[1].value * m2.get_shape()[2].value * m2.get_shape()[3].value
            flat = tf.reshape(m2, [-1, dim])

            fc = self._affine_map(flat, dim, n_hidden, "fc_layer", reuse=reuse)
            r3 = tf.nn.relu(fc)

            z_mean = self._affine_map(r3, n_hidden, z_dim, "mean_layer", reuse=reuse)

            a3_var = self._affine_map(r3, n_hidden, z_dim, "var_layer", reuse=reuse)
            z_var = tf.nn.softplus(a3_var)

            return z_mean, z_var, r3, [r1, m1_argmax, r2, m2_argmax]


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

            unpool_1 = self._unpool(Z_2d, self.m2_argmax, tf.shape(self.r2), "layer_3_unpool", reuse=reuse)
        
            r1 = tf.nn.relu(unpool_1)

            w1 = self._weight([3, 3, 16, 16], "layer_3", reuse=reuse)
            b1 = self._bias([16], "layer_3", reuse=reuse)
            c1 = tf.nn.conv2d_transpose(r1, w1, output_shape=[b_size,_h*2,_w*2,16], 
                strides=[1,1,1,1], padding='SAME') + b1

            unpool_2 = self._unpool(c1, self.m1_argmax, tf.shape(self.r1), "layer_4_unpool", reuse=reuse)

            r2 = tf.nn.relu(unpool_2)

            w2 = self._weight([3, 3, self._nc, 16], "layer_4", reuse=reuse)
            b2 = self._bias([self._nc], "layer_4", reuse=reuse)
            c2 = tf.nn.conv2d_transpose(r2, w2, output_shape=[b_size, self._h, self._w, self._nc], 
                strides=[1,1,1,1], padding='SAME') + b2

            x_logits = tf.reshape(c2, [-1, self._h * self._w * self._nc])
            x_probs = tf.nn.sigmoid(x_logits)

            return x_logits, x_probs

    
    def _weight(self, shape, scope, reuse):
        """
        Initialize weight variable.
        """
        with tf.variable_scope(scope, reuse=reuse):
            W = tf.get_variable("W", shape=shape, 
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            if not reuse:
                tf.summary.histogram("weight", W)
            
            return W


    def _bias(self, shape, scope, reuse):
        """
        Initialize bias variable. 
        """
        with tf.variable_scope(scope, reuse=reuse):
            b = tf.get_variable("b", shape=shape, initializer=tf.constant_initializer(0.1))
            return b


    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.stack(output_list)


    def _unpool(self, input, encoded_argmax, out_shape, scope, reuse):
        
        with tf.variable_scope(scope, reuse=reuse):
            argmax = self.unravel_argmax(encoded_argmax, tf.to_int64(out_shape))
            output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

            height = tf.shape(output)[0]
            width = tf.shape(output)[1]
            channels = tf.shape(output)[2]

            t1 = tf.to_int64(tf.range(channels))
            t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
            t1 = tf.reshape(t1, [-1, channels])
            t1 = tf.transpose(t1, perm=[1, 0])
            t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

            t2 = tf.squeeze(argmax)
            t2 = tf.stack((t2[0], t2[1]), axis=0)
            t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

            t = tf.concat([t2, t1], 3)
            indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

            x1 = tf.squeeze(input)
            x1 = tf.reshape(x1, [-1, channels])
            x1 = tf.transpose(x1, perm=[1, 0])
            values = tf.reshape(x1, [-1])

            delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
            return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)
