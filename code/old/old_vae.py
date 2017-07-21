import tensorflow as tf
from models import modules as mod
from models import base 


class VariationalAutoEncoder(base.Model):
    
    def __init__(self, input_dim, latent_dim, learning_rate, epsilon = 1e-3, decay = 0.99, 
        name="VAE", session=None, log_dir=None, model_dir=None):
        """
        Variational Auto-Encoder. 

        input_dim: input data dimensions
        latent_dim: latent space dimensionality
        learning_rate: optimization learning rate
        """
        # data and latent dimensionality
        self.x_dim = input_dim
        self.z_dim = latent_dim
        self.lr = learning_rate

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
            a1 = self._affine_map(self.X, self.x_dim, 500, "layer_1")
            h1 = tf.nn.relu(a1)
            a2 = self._affine_map(h1, 500, 500, "layer_2")
            h2 = tf.nn.relu(a2)

            z_mean = self._affine_map(h2, 500, self.z_dim, "mean_layer")

            a3_var = self._affine_map(h2, 500, self.z_dim, "var_layer")
            z_var = tf.nn.softplus(a3_var)

            return z_mean, z_var


    def _decoder(self,):
        """
        Generator network. 
        """
        with tf.variable_scope("decoder", reuse=False):
            a1 = self._affine_map(self.Z, self.z_dim, 500, "layer_1")
            h1 = tf.nn.relu(a1)
            a2 = self._affine_map(h1, 500, 500, "layer_2")
            h2 = tf.nn.relu(a2)

            x_logits = self._affine_map(h2, 500, self.x_dim, "layer_3")
            x_probs = tf.nn.sigmoid(x_logits)

            return x_logits, x_probs


    def _affine_map(self, input, in_dim, out_dim, scope, reuse):
        """
        Affine transform.

        input: input tensor
        in_dim/out_dim: input and output dimensions
        """
        with tf.variable_scope(scope, reuse=reuse):
            W = tf.get_variable("W", shape=[in_dim,out_dim], 
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            b = tf.get_variable("b", shape=[out_dim], initializer=tf.constant_initializer(0.1))

            return tf.matmul(input,W) + b

    
    def _sample_latent_space(self,):
        """
        Monte Carlo sampling from Latent distribution. Takes a single sample for each observation.
        """
        with tf.variable_scope("sampling", reuse=False):
            z_std = tf.sqrt(self.z_var)
            batch_size = tf.shape(self.X)[0]
            eps = tf.random_normal((batch_size, self.z_dim))
            z = self.z_mean + tf.multiply(z_std, eps)
            return z
    

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
            step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            return step


    def _summaries(self,):
        """
        Summary variables for visualizing with tensorboard.
        """
        with tf.variable_scope("summary", reuse=False):
            tf.summary.scalar('variational_bound', self.bound)
            return tf.summary.merge_all()


    def train(self, batch_X, write=True):
        """
        Executes single training step.

        batch_X: minibatch of input data
        write: indicates whether to write summary
        """
        feed = {self.X: batch_X}
        summary, _ = self.sess.run([self.summary, self.step], feed_dict=feed)
        if write:
            self.tr_writer.add_summary(summary, self.n_steps)
        self.n_steps = self.n_steps + 1
    

    def test(self, batch_X):
        """
        Writes summary for test data.

        batch_X: minibatch of input data
        """
        feed = {self.X: batch_X}
        loss, summary = self.sess.run([self.loss, self.summary], feed_dict=feed)
        self.te_writer.add_summary(summary, self.n_steps)

        return loss
        

    def reconstruct(self, batch_X):
        """
        Reconstructed data, given input X.

        batch_X: minibatch of input data
        """
        feed = {self.X: batch_X}
        return self.sess.run(self.x_probs, feed_dict=feed)

    
    def encode(self, batch_X):
        """
        Computes mean of latent space, given input X.

        batch_X: minibatch of input data        
        """
        feed = {self.X: batch_X}
        return self.sess.run(self.z_mean, feed_dict=feed)

    
    def decode(self, batch_Z):
        """
        Computes bernoulli probabilities in data space, given input Z.

        batch_Z: minibatch in latent space
        """
        feed = {self.Z: batch_Z}
        return self.sess.run(self.x_probs, feed_dict=feed)

