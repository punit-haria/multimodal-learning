import tensorflow as tf
from models import modules as mod
from models.joint_vae import JointVAE



class JointVAE_XtoY(JointVAE):
    
    def __init__(self, input_dim, latent_dim, learning_rate, n_hidden_units=200, joint_strategy='constrain',
        name="JointVAE", session=None, log_dir=None, model_dir=None):
        """
        Uses Txy + Lx for the joint variational bound.
        """
        super(JointVAE_XtoY, self).__init__(input_dim, latent_dim, learning_rate, n_hidden_units, 
            joint_strategy, name, session, log_dir, model_dir)


    def _joint_bound(self, scope='joint_bound'):
        with tf.variable_scope(scope, reuse=False):
            Txy = self._Txy()
            Lx = self._Lx()
            return tf.reduce_mean(Txy + Lx, axis=0)



class JointVAE_YtoX(JointVAE):
    
    def __init__(self, input_dim, latent_dim, learning_rate, n_hidden_units=200, joint_strategy='constrain',
        name="JointVAE", session=None, log_dir=None, model_dir=None):
        """
        Uses Tyx + Ly for the joint variational bound.
        """
        super(JointVAE_YtoX, self).__init__(input_dim, latent_dim, learning_rate, n_hidden_units, 
            joint_strategy, name, session, log_dir, model_dir)


    def _joint_bound(self, scope='joint_bound'):
        with tf.variable_scope(scope, reuse=False):
            Tyx = self._Tyx()
            Ly = self._Ly()
            return tf.reduce_mean(Tyx + Ly, axis=0)



class JointVAE_Average(JointVAE):
    
    def __init__(self, input_dim, latent_dim, learning_rate, n_hidden_units=200, joint_strategy='constrain',
        name="JointVAE", session=None, log_dir=None, model_dir=None):
        """
        Same as the JointVAE, but uses the average of Lxy, Txy + Lx, and Tyx + Ly for the joint
        variational bound.
        """
        super(JointVAE_Average, self).__init__(input_dim, latent_dim, learning_rate, n_hidden_units,  
            joint_strategy, name, session, log_dir, model_dir)


    def _joint_bound(self, scope='joint_bound'):
        with tf.variable_scope(scope, reuse=False):
            b1 = self._Txy() + self._Lx()
            b2 = self._Tyx() + self._Ly()
            b3 = self._Lxy()
            b = tf.stack([b1, b2, b3], axis=1)

            return tf.reduce_mean(b)
            


class JointVAE_Deeper(JointVAE):
    
    def __init__(self, input_dim, latent_dim, learning_rate, n_hidden_units=200, joint_strategy='constrain',
        name="JointVAE_Deeper", session=None, log_dir=None, model_dir=None):
        """
        Uses Txy + Lx for the joint variational bound.
        """
        super(JointVAE_Deeper, self).__init__(input_dim, latent_dim, learning_rate, n_hidden_units, 
            joint_strategy, name, session, log_dir, model_dir)


    def _q_z_x(self, X, x_dim, z_dim, n_hidden, scope, reuse):
        """
        Inference network.

        X: input data
        x_dim: dimensionality of input space 
        z_dim: dimensionality of latent space
        n_hidden: number of hidden units in each layer
        """
        with tf.variable_scope(scope, reuse=reuse):
            a1 = self._affine_map(X, x_dim, n_hidden, "layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)
            a2 = self._affine_map(h1, n_hidden, n_hidden, "layer_2", reuse=reuse)
            h2 = tf.nn.relu(a2)
            a3 = self._affine_map(h2, n_hidden, n_hidden, "layer_2", reuse=reuse)
            h3 = tf.nn.relu(a3)
            a4 = self._affine_map(h3, n_hidden, n_hidden, "layer_2", reuse=reuse)
            h4 = tf.nn.relu(a4)

            z_mean = self._affine_map(h4, n_hidden, z_dim, "mean_layer", reuse=reuse)

            a3_var = self._affine_map(h4, n_hidden, z_dim, "var_layer", reuse=reuse)
            z_var = tf.nn.softplus(a3_var)

            return z_mean, z_var, h4


    def _p_x_z(self, Z, z_dim, x_dim, n_hidden, scope, reuse):
        """
        Generator network. 

        Z: latent data
        z_dim: dimensionality of latent space
        x_dim: dimensionality of output space
        n_hidden: number of hidden units in each layer
        """
        with tf.variable_scope(scope, reuse=reuse):
            a1 = self._affine_map(Z, z_dim, n_hidden, "layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)
            a2 = self._affine_map(h1, n_hidden, n_hidden, "layer_2", reuse=reuse)
            h2 = tf.nn.relu(a2)
            a3 = self._affine_map(h2, n_hidden, n_hidden, "layer_2", reuse=reuse)
            h3 = tf.nn.relu(a3)
            a4 = self._affine_map(h3, n_hidden, n_hidden, "layer_2", reuse=reuse)
            h4 = tf.nn.relu(a4)

            x_logits = self._affine_map(h4, n_hidden, x_dim, "layer_3", reuse=reuse)
            x_probs = tf.nn.sigmoid(x_logits)

            return x_logits, x_probs
            
            