"""
Encoder networks.
"""
import tensorflow as tf 


def enc_1(output_dims):
	"""
	Encoder neural network with single hidden layer and affine transformed output. 
	"""
	with tf.variable_scope("enc_1", reuse=False):
		# input 
		input = tf.placeholder(tf.float32, [None, 784], name='input')

		# layer 1 weights
		W1 = tf.get_variable("W1", shape=[784,128], 
			initializer=tf.random_uniform_initializer(-0.1, 0.1), 
			dtype=tf.float32)
		b1 = tf.get_variable("b1", shape=[128], initializer=tf.constant_initializer(0.1))

		# affine transform 
		A1 = tf.matmul(input,W1) + b1

		# nonlinearity
		H1 = tf.nn.tanh(A1)

		# output layer weights
		W = tf.get_variable("W", shape=[128,output_dims], 
			initializer=tf.random_uniform_initializer(-0.1, 0.1), 
			dtype=tf.float32)
		b = tf.get_variable("b", shape=[output_dims], initializer=tf.constant_initializer(0.1))

		# final affine transform
		output = tf.matmul(H1,W) + b

	return input, output




