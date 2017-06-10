"""
Training functions.
"""
import tensorflow as tf 


def vae_loss(enc_in, enc_out, decoder, N, latent_dim):
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
        Mu = tf.slice(enc_out, begin=[0,0], size=[-1,latent_dim])       # means
        S = tf.slide(enc_out, begin=[0,latent_dim], size=[-1,1])        # standard deviations

        # monte carlo sampling of latents
        mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=Mu, scale_identity_multiplier=S)
        latents = mvn.sample()

        # decoder 
        dec_out = decoder(latents, latent_dim)

        # variational loss - part 1 
        l1 = tf.nn.softmax_cross_entropy_with_logits(logits=dec_out, labels=enc_in)
        L1 = tf.reduce_mean(l1, axis=0)
        
        # variational loss - part 2
        s1 = tf.mul(S,S)
        summ = tf.log(S) - tf.mul(0.5, s1)
        m1 = tf.reduce_sum(tf.mul(Mu,Mu), axis=1)
        l2 = tf.mul(J, tf.add(0.5, summ)) - m1
        L2 = tf.reduce_mean(l2, axis=0)

        # total loss
        loss = tf.multiply(L1 + L2, N)

        # summaries
        tf.summary.scalar('loss', loss)

    return latents, dec, loss
