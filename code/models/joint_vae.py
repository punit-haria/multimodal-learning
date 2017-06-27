import tensorflow as tf
from models import modules as mod
from models import base 


class JointVAE(base.Model):
    
    def __init__(self, input_dim, latent_dim, learning_rate, epsilon = 1e-3, decay = 0.99, 
        name="VAE", session=None, log_dir=None, model_dir=None):
        """
        Joint Variational Auto-Encoder (Hippolyt Ritter)

        input_dim: tuple of output variable dimensions (assumes 2 output variables)
        latent_dim: latent space dimensionality
        learning_rate: optimization learning rate
        """
        # data and latent dimensionality
        self.x_dim, self.y_dim = input_dim
        self.z_dim = latent_dim
        self.lr = learning_rate

        # counter for number of executed training steps
        self.n_steps = 0

        # base class constructor (initializes model)
        super(JointVAE, self).__init__(name=name, session=session, 
            log_dir=log_dir, model_dir=model_dir)


    def _initialize(self,):
        """
        Initialize model components. 
        """
        with tf.variable_scope(self.name, reuse=False):  
            
            # joint-input variables
            self.X_joint = tf.placeholder(tf.float32, [None, self.x_dim], name='X_joint')
            self.Y_joint = tf.placeholder(tf.float32, [None, self.y_dim], name='Y_joint')

            # X input with missing Y
            self.X = tf.placeholder(tf.float32, [None, self.x_dim], name='X')

            # Y input with missing X
            self.Y = tf.placeholder(tf.float32, [None, self.y_dim], name='Y')

            # q(z|x) parameters
            self.zx_mean, self.zx_var = self._q_z_x(self.X, self.x_dim, self.z_dim, 
                scope='q_z_x', reuse=False)
            # q(z|y) parameters 
            self.zy_mean, self.zy_var = self._q_z_x(self.Y, self.y_dim, self.z_dim, 
                scope='q_z_y', reuse=False)

            # q(z|x,y) parameters (using product of two Gaussians)
            zxm, zxv = self._q_z_x(self.X_joint, self.x_dim, self.z_dim,
                scope='q_z_x', reuse=True)
            zym, zyv = self._q_z_x(self.Y_joint, self.y_dim, self.z_dim,
                scope='q_z_y', reuse=True)
            zxv_inv = tf.reciprocal(zxv)
            zyv_inv = tf.reciprocal(zyv)
            self.zxy_var = tf.reciprocal(zxv_inv + zyv_inv)
            zx = tf.multiply(zxv_inv, zxm)
            zy = tf.multiply(zyv_inv, zym)
            self.zxy_mean = tf.multiply(self.zxy_var, zx + zy)

            # batch sizes for each case
            zx_samples = tf.shape(self.X)[0]
            zy_samples = tf.shape(self.Y)[0]
            zxy_samples = tf.shape(self.X_joint)[0]

            # sampling 
            self.Zx = self._sample_latent_space(self.zx_mean, self.zx_var, self.z_dim, zx_samples)
            self.Zy = self._sample_latent_space(self.zy_mean, self.zy_var, self.z_dim, zy_samples)
            self.Zxy = self._sample_latent_space(self.zxy_mean, self.zxy_var, self.z_dim, zxy_samples)

            # p(x|z) parameters
            self.xy_logits, self.xy_probs = self._p_x_z(self.Zy, self.z_dim, self.x_dim,
                scope='p_x_z', reuse=False)
            self.xx_logits, self.xx_probs = self._p_x_z(self.Zx, self.z_dim, self.x_dim,
                scope='p_x_z', reuse=True)
            
            # p(y|z) parameters
            self.yx_logits, self.yx_probs = self._p_x_z(self.Zx, self.z_dim, self.y_dim,
                scope='p_y_z', reuse=False)
            self.yy_logits, self.yy_probs = self._p_x_z(self.Zy, self.z_dim, self.y_dim,
                scope='p_y_z', reuse=True)
            
            # joint p(x|z) and p(y|z) parameters
            self.x_logits_joint, self.x_probs_joint = self._p_x_z(self.Zxy, self.z_dim, self.x_dim,
                scope='p_x_z', reuse=True)
            self.y_logits_joint, self.y_probs_joint = self._p_x_z(self.Zxy, self.z_dim, self.y_dim,
                scope='p_y_z', reuse=True)

            # bound: X observed, Y missing
            marginal_x = self._variational_bound(self.xx_logits, self.X, self.zx_mean, self.zx_var,
                scope='marginal_x')
            #l1_p_y_z = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=self.yx_logits, 
            #    labels=self.Y), axis=1)
            #x_bound = tf.reduce_mean(marginal_x + l1_p_y_z, axis=0)
            x_bound = tf.reduce_mean(marginal_x, axis=0)
            
            # bound: Y observed, X missing
            marginal_y = self._variational_bound(self.yy_logits, self.Y, self.zy_mean, self.zy_var,
                scope='marginal_y')
            #l1_p_x_z = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_logits, 
            #    labels=self.X), axis=1)
            #y_bound = tf.reduce_mean(marginal_y + l1_p_x_z, axis=0)
            y_bound = tf.reduce_mean(marginal_y, axis=0)

            # bound: X,Y observed
            l1_x = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_logits_joint, 
                labels=self.X_joint), axis=1)
            l1_y = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_logits_joint, 
                labels=self.Y_joint), axis=1)
            l2 = 0.5 * tf.reduce_sum(1 + tf.log(self.zxy_var) - tf.square(self.zxy_mean) - self.zxy_var, axis=1)
            joint_bound = tf.reduce_mean(l1_x + l1_y + l2, axis=0)

            # summaries
            tf.summary.scalar('x_bound', x_bound)
            tf.summary.scalar('y_bound', y_bound)
            tf.summary.scalar('joint_bound', joint_bound)

            # final bound
            self.bound = x_bound + y_bound + joint_bound

            # loss
            self.loss = -self.bound

            # optimization step
            self.step = self._optimizer()


    def _p_x_z(self, Z, z_dim, x_dim, scope, reuse):
        """
        Generator network. 

        Z: latent data
        z_dim: dimensionality of latent space
        x_dim: dimensionality of output space 
        """
        with tf.variable_scope(scope, reuse=reuse):
            a1 = mod.affine_map(Z, z_dim, 200, "layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)
            a2 = mod.affine_map(h1, 200, 200, "layer_2", reuse=reuse)
            h2 = tf.nn.relu(a2)

            x_logits = mod.affine_map(h2, 200, x_dim, "layer_3", reuse=reuse)
            x_probs = tf.nn.sigmoid(x_logits)

            return x_logits, x_probs

    
    def _q_z_x(self, X, x_dim, z_dim, scope, reuse):
        """
        Inference network.

        X: input data
        x_dim: dimensionality of input space 
        z_dim: dimensionality of latent space
        """
        with tf.variable_scope(scope, reuse=reuse):
            a1 = mod.affine_map(X, x_dim, 200, "layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)
            a2 = mod.affine_map(h1, 200, 200, "layer_2", reuse=reuse)
            h2 = tf.nn.relu(a2)

            z_mean = mod.affine_map(h2, 200, z_dim, "mean_layer", reuse=reuse)

            a3_var = mod.affine_map(h2, 200, z_dim, "var_layer", reuse=reuse)
            z_var = tf.nn.softplus(a3_var)

            return z_mean, z_var


    def _sample_latent_space(self, z_mean, z_var, z_dim, n_samples, scope='sampling', reuse=False):
        """
        Monte Carlo sampling from Latent distribution. Takes a single sample for each observation.
        """
        with tf.variable_scope(scope, reuse=reuse):
            z_std = tf.sqrt(z_var)
            eps = tf.random_normal((n_samples, z_dim))
            z = z_mean + tf.multiply(z_std, eps)
            return z
    

    def _variational_bound(self, logits, labels, mean, var, scope='variational_bound'):
        """
        Variational Bound.
        """
        with tf.variable_scope(scope, reuse=False):
            # reconstruction
            l1 = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, 
                labels=labels), axis=1)

            # penalty
            l2 = 0.5 * tf.reduce_sum(1 + tf.log(var) - tf.square(mean) - var, axis=1)

            # total bound (unreduced)
            return l1 + l2
        

    def _optimizer(self,):
        """
        Optimization method.
        """
        with tf.variable_scope("optimization", reuse=False):
            step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            return step


    def _summaries(self,):
        """
        Summary variables for visualizing with tensorboard.
        """
        with tf.variable_scope("summary", reuse=False):
            #tf.summary.scalar('variational_bound', self.bound)
            return tf.summary.merge_all()


    def train(self, X, Y, X_joint, Y_joint, write=True):
        """
        Executes single training step.

        write: indicates whether to write summary
        """
        feed = {self.X: X, self.Y: Y, self.X_joint: X_joint, self.Y_joint: Y_joint}
        summary, _ = self.sess.run([self.summary, self.step], feed_dict=feed)
        if write:
            self.tr_writer.add_summary(summary, self.n_steps)
        self.n_steps = self.n_steps + 1
    

    def test(self, X, Y, X_joint, Y_joint):
        """
        Writes summary for test data.
        """
        feed = {self.X: X, self.Y: Y, self.X_joint: X_joint, self.Y_joint: Y_joint}
        loss, summary = self.sess.run([self.loss, self.summary], feed_dict=feed)
        self.te_writer.add_summary(summary, self.n_steps)

        return loss
        

    def reconstruct(self, X, Y, X_joint, Y_joint):
        """
        Reconstructed data, given input X.
        """
        feed = {self.X: X, self.Y: Y, self.X_joint: X_joint, self.Y_joint: Y_joint}
        return self.sess.run(self.x_probs, feed_dict=feed)

    
    def encode(self, X, Y, X_joint, Y_joint):
        """
        Computes mean of latent space, given input X.  
        """
        feed = {self.X: X, self.Y: Y, self.X_joint: X_joint, self.Y_joint: Y_joint}
        return self.sess.run(self.zx_mean, feed_dict=feed)

    
    def decode(self, X, Y, X_joint, Y_joint):
        """
        Computes bernoulli probabilities in data space, given input Z.
        """
        feed = {self.X: X, self.Y: Y, self.X_joint: X_joint, self.Y_joint: Y_joint}
        return self.sess.run(self.x_probs, feed_dict=feed)

