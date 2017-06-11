"""
Training functions.
"""
import tensorflow as tf 


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
        dec_out = decoder(latents, latent_dim)

        # variational loss - part 1 
        l1 = tf.nn.softmax_cross_entropy_with_logits(logits=dec_out, labels=enc_in)
        L1 = tf.reduce_mean(l1, axis=0)
        
        # variational loss - part 2
        J = float(latent_dim)
        s1 = tf.multiply(S,S)
        summ = tf.multiply(0.5, tf.log(s1)) - tf.multiply(0.5, s1)
        m1 = tf.reduce_sum(tf.multiply(Mu,Mu), axis=1)
        l2 = tf.multiply(J, tf.add(0.5, summ)) - m1
        L2 = tf.reduce_mean(l2, axis=0)

        # total loss
        bound = tf.multiply(L1 + L2, N)

    return latents, dec_out, bound



def generate(decoder, n_images, latent_dim):
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
    gen_ims = sess.run(dec, feed_dict={latents:Z})

    # reshape
    gen_ims = np.reshape(gen_ims, [-1,28,28])

    return gen_ims


