import tensorflow as tf
import numpy as np 

import data
import encoders as enc
import decoders as dec
import loss

# parameters
n_steps = 1000                        # number of training steps
batch_size = 100                    # minibatch size
latent_dim = 20                     # dimensionality of latent variable
n_parms = latent_dim + 1            # number of parameters in q(z|x) distribution (Gaussian)
lr = 0.001                          # learning rate

# train and test sets
Xtr, ytr = data.mnist('train')
Xte, yte = data.mnist('test')

# size of datasets
Ntr = len(Xtr)
Nte = len(Xte)

# create and initialize Session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# encoder network
enc_input, enc_output = enc.enc_1(n_parms)

# variational loss
latents, dec, loss = loss.vae_loss(enc_input, enc_output, dec.dec_1, Ntr, latent_dim)

# optimizer
step = tf.train.AdamOptimizer(lr).minimize(loss)

# tf summaries
merged = tf.summary.merge_all()

# summary writer
train_writer = tf.train.SummaryWriter('logs/train_'+model.name) 

# train model
for i in range(n_steps):
    
    # randomly sampled minibatch 
    idx = np.random.randint(Ntr, size=batch_size)
    Xb = Xtr[idx,:]

    # optimize
    _, summary = sess.run([step, merged], feed_dict={enc_input: Xb})

    # save summary 












	# random batch sample --> stochastic gradient descent
	batch = mnist.train.next_batch(batch_size)
	# training step 
	_, summary = sess.run([model.step, model.merged], feed_dict={model.X: batch[0], model.y: batch[1]})
	train_writer.add_summary(summary, i)

	if i % 100 == 0:
		train_err = sess.run(model.error, feed_dict={model.X: batch[0], model.y: batch[1]})
		test_err, summary = sess.run([model.error, model.merged], feed_dict={model.X: mnist.test.images, model.y: mnist.test.labels})
		test_writer.add_summary(summary, i)
		print("Training error: ", train_err)
		print("Validation error: ", test_err)
