"""
Encoder networks.
"""
import tensorflow as tf 


def enc_1(latent_dims):
	"""
	Encoder neural network with single hidden layer and affine transformed output. 
	"""
	with tf.variable_scope("enc_1", reuse=False):
		# input 
		x = tf.placeholder(tf.float32, [None, 784], name='input')

		# layer 1 weights
		W1 = tf.get_variable("W1", shape=[784,128], 
			initializer=tf.random_uniform_initializer(-0.1, 0.1), 
			dtype=tf.float32)
		b1 = tf.get_variable("b1", shape=[128], initializer=tf.constant_initializer(0.1))

		# affine transform 
		A1 = tf.matmul(x,W1) + b1

		# nonlinearity
		H1 = tf.nn.tanh(A1)

		# mean layer weights
		Wm = tf.get_variable("W_mean", shape=[128,latent_dims], 
			initializer=tf.random_uniform_initializer(-0.1, 0.1), 
			dtype=tf.float32)
		bm = tf.get_variable("b_mean", shape=[latent_dims], initializer=tf.constant_initializer(0.1))

		Wlv = tf.get_variable("W_logvar", shape=[128,latent_dims], 
			initializer=tf.random_uniform_initializer(-0.1, 0.1), 
			dtype=tf.float32)
		blv = tf.get_variable("b_logvar", shape=[latent_dims], initializer=tf.constant_initializer(0.1))

		# final affine transforms
		z_mean = tf.matmul(H1,Wm) + bm
		z_logvar = tf.matmul(H1,Wlv) + blv

	return x, z_mean, z_logvar




