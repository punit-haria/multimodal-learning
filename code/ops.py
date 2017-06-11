"""
Training functions.
"""
import tensorflow as tf 
import numpy as np

from scipy.stats import norm


def vae_bound(enc_in, enc_out, decoder, N, latent_dim):
    """
    AEVB loss- Variational lower bound using MC estimate for expectation-gradients.

    enc_in: encoder input (data)
    enc_out: encoder output (mean and variances)
    decoder: decoder function
    N: total size of training set
    latent_dim: dimensionality of latent variable
    """
    with tf.variable_scope("vae_loss", reuse=False):

        # q() parameters
        Mu = tf.slice(enc_out, begin=[0,0], size=[-1,latent_dim])                   # means
        S = tf.squeeze(tf.slice(enc_out, begin=[0,latent_dim], size=[-1,1]))    # standard deviations

        # monte carlo sampling of latents
        mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=Mu, scale_identity_multiplier=S)
        latents = tf.squeeze(mvn.sample())

        # decoder 
        dec_logits, dec_probs = decoder(latents, latent_dim)

        # variational loss - reconstruction
        l1 = -tf.nn.softmax_cross_entropy_with_logits(logits=dec_logits, labels=enc_in)
        
        # variational loss - penalty
        J = float(latent_dim)
        s1 = tf.multiply(S,S)
        m1 = tf.reduce_sum(tf.multiply(Mu,Mu), axis=1)
        l2 = J*(0.5 + (0.5*(tf.log(s1) - s1))) - m1

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


def encode_mean(enc_input, enc_output, X, sess):
    """
    Encodes input X to latent space. Returns mean of q() distribution. 

    enc_input: encoder network input
    enc_output: encoder network output
    X: input matrix 
    """
    Q = sess.run(enc_output, feed_dict={enc_input: X})
    means = Q[:,:-1]  # encoded mean parameters

    return means



