"""
Decoder networks.
"""
import tensorflow as tf 

def dec_1(input, input_dims):
	"""
	Decoder neural network with single hidden layer and logit output. 

	input: input tf.variable
	input_dims: dimensionality of latent variable
	"""
	with tf.variable_scope("dec_1", reuse=False):

		# layer 1 weights
		W1 = tf.get_variable("W1", shape=[input_dims,128], 
			initializer=tf.random_uniform_initializer(-0.1, 0.1), 
			dtype=tf.float32)
		b1 = tf.get_variable("b1", shape=[128], initializer=tf.constant_initializer(0.1))

		# affine transform 
		A1 = tf.matmul(input,W1) + b1

		# nonlinearity
		H1 = tf.nn.tanh(A1)

		# output layer weights
		W = tf.get_variable("W", shape=[128,784], 
			initializer=tf.random_uniform_initializer(-0.1, 0.1), 
			dtype=tf.float32)
		b = tf.get_variable("b", shape=[784], initializer=tf.constant_initializer(0.1))

		# output as logits
		output = tf.matmul(H1,W) + b

		# softmaxed output
		probs = tf.nn.softmax(output)

	return output, probs

