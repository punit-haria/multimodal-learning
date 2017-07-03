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

            return tf.reduce_mean(b1 + b2 + b3)
            

            
            