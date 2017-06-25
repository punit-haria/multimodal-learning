import tensorflow as tf
import modules as mod
import base 


class VariationalAutoEncoder(base.Model):
    
    def __init__(self, input_dim, latent_dim, learning_rate, epsilon = 1e-3, decay = 0.99, 
        name="VAE", session=None, log_dir=None, model_dir=None):
        """
        Variational Auto-Encoder. 

        input_dim: input data dimensions
        latent_dim: latent space dimensionality
        learning_rate: optimization learning rate
        epsilon/decay: batch normalization parameters
        """
        # data and latent dimensionality
        self.x_dim = input_dim
        self.z_dim = latent_dim
        self.lr = learning_rate
        self.eps = epsilon
        self.decay = decay

        # counter for number of executed training steps
        self.n_steps = 0

        # base class constructor (initializes model)
        super(VariationalAutoEncoder, self).__init__(name=name, session=session, 
            log_dir=log_dir, model_dir=model_dir)


    def _initialize(self,):
        """
        Initialize model components. 
        """
        with tf.variable_scope(self.name, reuse=False):  
            # training indicator
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            # input placeholder
            self.X = tf.placeholder(tf.float32, [None, self.x_dim], name='X')

            # latent space parameters
            self.z_mean, self.z_var = self._encoder()

            # samples from latent space
            self.Z = self._sample_latent_space()

            # model parameters
            self.x_logits, self.x_probs = self._decoder()      

            # variational bound
            self.bound = self._variational_bound()

            # loss
            self.loss = -self.bound  

            # optimization step
            self.step = self._optimizer()


    def _encoder(self,):
        """
        Recognition network.
        """
        with tf.variable_scope("encoder", reuse=False):
            a1 = mod.affine_map(self.X, self.x_dim, 500, "layer_1")
            b1 = mod.batch_norm(a1, 'b1', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h1 = tf.nn.relu(b1)
            a2 = mod.affine_map(h1, 500, 500, "layer_2")
            b2 = mod.batch_norm(a2, 'b2', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h2 = tf.nn.relu(b2)

            a3_mean = mod.affine_map(h2, 500, self.z_dim, "mean_layer")
            z_mean = mod.batch_norm(a3_mean, 'z_mean', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False) 

            a3_var = mod.affine_map(h2, 500, self.z_dim, "var_layer")
            b3_var = mod.batch_norm(a3_var, 'b3_var', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False) 
            z_var = tf.nn.softplus(b3_var)

            return z_mean, z_var


    def _decoder(self,):
        """
        Generator network. 
        """
        with tf.variable_scope("decoder", reuse=False):
            a1 = mod.affine_map(self.Z, self.z_dim, 500, "layer_1")
            b1 = mod.batch_norm(a1, 'b1', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h1 = tf.nn.relu(b1)
            a2 = mod.affine_map(h1, 500, 500, "layer_2")
            b2 = mod.batch_norm(a2, 'b2', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h2 = tf.nn.relu(b2)

            a3 = mod.affine_map(h2, 500, self.x_dim, "layer_3")
            x_logits = mod.batch_norm(a3, 'x_logits', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            x_probs = tf.nn.sigmoid(x_logits)

            return x_logits, x_probs

    
    def _sample_latent_space(self,):
        """
        Monte Carlo sampling from Latent distribution. Takes a single sample for each observation.
        """
        with tf.variable_scope("sampling", reuse=False):
            z_std = tf.sqrt(self.z_var)
            mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=self.z_mean, scale_diag=z_std)
            return tf.squeeze(mvn.sample())
    

    def _variational_bound(self,):
        """
        Variational Bound.
        """
        with tf.variable_scope("variational_bound", reuse=False):
            # reconstruction
            l1 = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_logits, 
                labels=self.X), axis=1)

            # penalty
            l2 = 0.5 * tf.reduce_sum(1 + tf.log(self.z_var) - tf.square(self.z_mean) - self.z_var, axis=1)

            # total bound
            return tf.reduce_mean(l1+l2, axis=0)
        

    def _optimizer(self,):
        """
        Optimization method.
        """
        with tf.variable_scope("optimization", reuse=False):
            step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            return step


    def _summaries(self,):
        """
        Summary variables for visualizing with tensorboard.
        """
        with tf.variable_scope("summary", reuse=False):
            tf.summary.scalar('variational_bound', self.bound)
            #tf.summary.histogram('z_mean', self.z_mean)
            #tf.summary.histogram('z_var', self.z_var)
            return tf.summary.merge_all()


    def train(self, batch_X, write=True):
        """
        Executes single training step.

        batch_X: minibatch of input data
        write: indicates whether to write summary
        """
        feed = {self.X: batch_X, self.is_training: True}
        summary, _ = self.sess.run([self.summary, self.step], feed_dict=feed)
        if write:
            self.tr_writer.add_summary(summary, self.n_steps)
        self.n_steps = self.n_steps + 1
    

    def test(self, batch_X):
        """
        Writes summary for test data.

        batch_X: minibatch of input data
        """
        feed = {self.X: batch_X, self.is_training: False}
        loss, summary = self.sess.run([self.loss, self.summary], feed_dict=feed)
        self.te_writer.add_summary(summary, self.n_steps)

        return loss
        

    def reconstruct(self, batch_X):
        """
        Reconstructed data, given input X.

        batch_X: minibatch of input data
        """
        feed = {self.X: batch_X, self.is_training: False}
        return self.sess.run(self.x_probs, feed_dict=feed)

    
    def encode(self, batch_X):
        """
        Computes mean of latent space, given input X.

        batch_X: minibatch of input data        
        """
        feed = {self.X: batch_X, self.is_training: False}
        return self.sess.run(self.z_mean, feed_dict=feed)

    
    def decode(self, batch_Z):
        """
        Computes bernoulli probabilities in data space, given input Z.

        batch_Z: minibatch in latent space
        """
        feed = {self.Z: batch_Z, self.is_training: False}
        return self.sess.run(self.x_probs, feed_dict=feed)

