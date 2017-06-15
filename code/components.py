import tensorflow as tf
import os


class VariationalAutoEncoder(object):

    def __init__(self, input_dim, latent_dim, learning_rate, model_name,
        log_dir='../logs/', model_dir='../models/'):
        """
        Variational Auto-Encoder. 

        input_dim: input data dimensions
        latent_dim: latent space dimensionality
        learning_rate: optimization learning rate
        model_name: identifier for current experiment
        """
        # data and latent dimensionality
        self.x_dim = input_dim
        self.z_dim = latent_dim
        self.lr = learning_rate
        self.name = model_name

        # input placeholder
        self.X = tf.placeholder(tf.float32, [None, self.x_dim], name='X')

        # latent space parameters
        self.z_mean, self.z_std = self._encoder()

        # samples from latent space
        self.Z = self._sample_latent_space()

        # model parameters
        self.x_logits, self.x_probs = self._decoder()      

        # loss
        self.loss = self._variational_loss()

        # optimization step
        self.step = self._optimizer()

        # summary variables 
        self.summary = self._summaries()

        # logging and saved model directories
        self.log_dir = log_dir
        self.model_dir = model_dir

        # summary writers
        self.tr_writer = tf.summary.FileWriter(self.log_dir+self.name+'_train_') 
        self.te_writer = tf.summary.FileWriter(self.log_dir+self.name+'_test_') 

        # counts number of executed training steps
        self.n_steps = 0

        # new tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
          

    def _encoder(self,):
        """
        Recognition network.
        """
        with tf.variable_scope("encoder", reuse=False):
            h1 = tf.nn.tanh(affine_map(self.X, self.x_dim, 500, "layer_1"))
            h2 = tf.nn.tanh(affine_map(h1, 500, 500, "layer_2"))

            z_mean = affine_map(h2, 500, self.z_dim, "z_mean")
            z_std = tf.nn.softplus(affine_map(h2, 500, self.z_dim, "z_std"))

            return z_mean, z_std


    def _decoder(self,):
        """
        Generator network. 
        """
        with tf.variable_scope("decoder", reuse=False):
            h1 = tf.nn.tanh(affine_map(self.Z, self.z_dim, 500, "layer_1"))
            h2 = tf.nn.tanh(affine_map(h1, 500, 500, "layer_2"))

            x_logits = affine_map(h2, 500, self.x_dim, "x_logits")
            x_probs = tf.nn.sigmoid(x_logits)

            return x_logits, x_probs

    
    def _sample_latent_space(self,):
        """
        Monte Carlo sampling from Latent distribution. Takes a single sample for each observation.
        """
        with tf.variable_scope("sampling", reuse=False):
            mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=self.z_mean, scale_diag=self.z_std)
            return tf.squeeze(mvn.sample())
    

    def _variational_loss(self,):
        """
        Negative evidence lower bound.
        """
        with tf.variable_scope("variational_bound", reuse=False):
            # negative reconstruction 
            l1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_logits, 
                labels=self.X))
        
            # penalty
            l2 = -0.5 * tf.reduce_sum(1 + tf.log(tf.square(self.z_std)) 
                - tf.square(self.z_mean) - tf.square(self.z_std))

            return l1+l2


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
            tf.summary.scalar('variational loss', self.loss)
            return tf.summary.merge_all()


    def train(self, batch_X, write=True):
        """
        Executes single training step.

        batch_X: minibatch of input data
        write: indicates whether to write summary
        """
        summary, _ = self.sess.run([self.summary, self.step], feed_dict={self.X: batch_X})
        if write:
            self.tr_writer.add_summary(summary, self.n_steps)
        self.n_steps = self.n_steps + 1
    

    def test(self, batch_X):
        """
        Writes summary for test data.

        batch_X: minibatch of input data
        """
        summary = self.sess.run(self.summary, feed_dict={self.X: batch_X})
        self.te_writer.add_summary(summary, self.n_steps)
        

    def reconstruct(self, batch_X):
        """
        Reconstructed data, given input X.

        batch_X: minibatch of input data
        """
        return self.sess.run(self.x_probs, feed_dict={self.X: batch_X})

    
    def encode(self, batch_X):
        """
        Computes mean of latent space, given input X.

        batch_X: minibatch of input data        
        """
        return self.sess.run(self.z_mean, feed_dict={self.X: batch_X})

    
    def decode(self, batch_Z):
        """
        Computes bernoulli probabilities in data space, given input Z.

        batch_Z: minibatch in latent space
        """
        return self.sess.run(self.x_probs, feed_dict={self.Z: batch_Z})


    def save_state(self, name=None):
        """
        Save model.
        """
        if name is None:
            name = self.name
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        saver = tf.train.Saver()
        saver.save(self.sess, self.model_dir+name)


    def load_state(self, name=None):
        """
        Load model.
        """
        if name is None:
            name = self.name
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_dir+name)




def affine_map(input, in_dim, out_dim, scope):
    """
    Affine transform.

    input: input tensor
    in_dim/out_dim: input and output dimensions
    scope: variable scope as string
    """
    with tf.variable_scope(scope, reuse=False):
        W = tf.get_variable("W", shape=[in_dim,out_dim], 
			initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
        b = tf.get_variable("b", shape=[out_dim], initializer=tf.constant_initializer(0))

        return tf.matmul(input,W) + b


def batch_norm(x, scope, decay, epsilon, is_training, center=True):
	"""
	Batch normalization layer

  	This was implemented while referring to the following papers/resources:
  	https://arxiv.org/pdf/1502.03167.pdf
  	https://arxiv.org/pdf/1603.09025.pdf
  	http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
	"""
	with tf.variable_scope(scope):
		# number of features in input matrix
		dim = x.get_shape()[1].value
		# scaling coefficients
		gamma = tf.get_variable('gamma', [dim], initializer=tf.constant_initializer(1.0), trainable=True)
		# offset coefficients (if required)
		if center:
			beta = tf.get_variable('beta', [dim], initializer=tf.constant_initializer(0), trainable=True)
		else:
			beta = None

		# population mean variable (for prediction)
		popmean = tf.get_variable('pop_mean', shape=[dim], initializer=tf.constant_initializer(0.0), trainable=False)
		# population variance variable (for prediction)
		popvar = tf.get_variable('pop_var', shape=[dim], initializer=tf.constant_initializer(1.0), trainable=False)

		# compute batch mean and variance
		batch_mean, batch_var = tf.nn.moments(x, axes=[0])

		def update_and_train():
			# update popmean and popvar using moving average of batch_mean and batch_var
			pop_mean_new = popmean * decay + batch_mean * (1 - decay)
			pop_var_new = popvar * decay + batch_var * (1 - decay)
			with tf.control_dependencies([popmean.assign(pop_mean_new), popvar.assign(pop_var_new)]):
				# batch normalization
		  		return tf.nn.batch_normalization(x, mean=batch_mean, variance=batch_var, 
		  			offset=beta, scale=gamma, variance_epsilon=epsilon)

		def predict():
			# batch normalization (using population moments)
			return tf.nn.batch_normalization(x, mean=popmean, variance=popvar, 
				offset=beta, scale=gamma, variance_epsilon=epsilon)

		# conditional evaluation in tf graph
		return tf.cond(is_training, update_and_train, predict)

