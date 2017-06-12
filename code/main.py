import tensorflow as tf
import numpy as np 

import data
import encoders as enc
import decoders as dec
import ops 
import plot

# parameters
lr = 0.0001                             # learning rate
batch_size = 100                        # minibatch size
latent_dim = 2                          # dimensionality of latent variable
n_parms = latent_dim + 1                # number of parameters in q(z|x) distribution (Gaussian)
n_steps = 3000                         # number of training steps


# train and test sets
Xtr, ytr = data.mnist('train')
Xte, yte = data.mnist('test')

# size of datasets
Ntr = len(Xtr)
Nte = len(Xte)

# encoder network
enc_input, enc_output = enc.enc_1(n_parms)

# variational loss
latents, dec_out, dec_probs, bound = ops.vae_bound(enc_input, enc_output, dec.dec_1, Ntr, latent_dim)

# optimizer
step = tf.train.AdamOptimizer(lr).minimize(-bound)

# tf summaries
tf.summary.scalar('variational lower bound', bound)
merged = tf.summary.merge_all()

# summary writer
train_writer = tf.summary.FileWriter('../logs/train_vae') 
test_writer = tf.summary.FileWriter('../logs/test_vae')


# create and initialize Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train model
for i in range(n_steps):
    
    # randomly sampled minibatch 
    idx = np.random.randint(Ntr, size=batch_size)
    Xb = Xtr[idx,:]

    # optimize
    _, summary = sess.run([step, merged], feed_dict={enc_input: Xb})

    # save summary 
    train_writer.add_summary(summary, i)

    if i % 200 == 0:
        print("At iteration ", i)
        summary = sess.run(merged, feed_dict={enc_input: Xte})
        test_writer.add_summary(summary, i)

        # generate images
        #images = ops.generate_random(dec_probs, 25, latents, latent_dim, sess)
        images = ops.generate_uniform(dec_probs, 20, latents, sess)
        plot.plot_images(images, 20, 20, '../plots/images_'+str(i))

        # visualize latent space
        Zmu = ops.encode_mean(enc_input, enc_output, Xte, sess)
        plot.plot_latent_space(Zmu, yte, '../plots/latent_'+str(i))

        # generate reconstructions
        n_recons = 8
        idx = np.random.randint(Nte, size=n_recons)
        Xtb = Xte[idx,:]
        R = sess.run(dec_probs, feed_dict={enc_input: Xtb})
        images = np.concatenate((R, Xtb), axis=0)
        images = np.reshape(images, [-1,28,28])
        plot.plot_images(images, 4, 4, '../plots/reconstructions_'+str(i))


