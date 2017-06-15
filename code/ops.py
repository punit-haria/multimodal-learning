"""
Training functions.
"""
import tensorflow as tf 
import numpy as np

from scipy.stats import norm


def vae_bound(x, z_mean, z_logvar, decoder, N, latent_dim):
    """
    AEVB loss- Variational lower bound using MC estimate for expectation-gradients.

    x: encoder input (data)
    z_mean, z_logvar: encoder output (mean and variances)
    decoder: decoder function
    N: total size of training set
    latent_dim: dimensionality of latent variable
    """
    with tf.variable_scope("vae_loss", reuse=False):
        
        # q() standard deviation
        stdev = tf.sqrt(tf.exp(z_logvar))

        # monte carlo sampling of latents
        mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=stdev)
        latents = tf.squeeze(mvn.sample())

        # decoder 
        dec_logits, dec_probs = decoder(latents, latent_dim)

        # variational loss - reconstruction
        l1 = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=dec_logits, labels=x), axis=1)
        
        # variational loss - penalty
        l2 = 0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)

        # total loss
        bound = N * tf.reduce_mean(l1+l2, axis=0)

    return latents, dec_logits, dec_probs, bound



def generate_random(decoder, n_images, latents, latent_dim, sess):
    """
    Generates random images from decoder network.

    decoder: decoder network output
    n_images: number of images
    latent_dim: dimensionality of latent variable z
    """
    # randomly sampled latents from prior distribution N(0,I)
    Z = np.random.multivariate_normal(mean=np.zeros(latent_dim),
        cov=np.identity(latent_dim),
        size=n_images)
    
    # generate bernoulli probabilities from decoder
    gen_ims = sess.run(decoder, feed_dict={latents:Z})

    # reshape
    gen_ims = np.reshape(gen_ims, [-1,28,28])

    return gen_ims


def generate_uniform(decoder, n, latents, sess):
    """
    Generates uniformly spaced images from decoder network. Assumes latent dimensionality is 2.

    decoder: decoder network output
    n: number of images in each row/column
    latent_dim: dimensionality of latent variable z
    """
    # 2D grid of evenly spaced latents
    z1 = np.linspace(-3, 3, n)
    z2 = np.linspace(-3, 3, n)

    # inverse CDF transform
    Z = []
    for i, x in enumerate(z1):
        for j, y in enumerate(z2):
            Z.append([x,y])
    Z = np.array(Z)
    
    # generate bernoulli probabilities from decoder
    gen_ims = sess.run(decoder, feed_dict={latents:Z})

    # reshape
    gen_ims = np.reshape(gen_ims, [-1,28,28])

    return gen_ims

