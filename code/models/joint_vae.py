import tensorflow as tf
from models import modules as mod
from models import base 


class JointVAE(base.Model):
    
    def __init__(self, input_dim, latent_dim, learning_rate, n_hidden_units=200,
        name="VAE", session=None, log_dir=None, model_dir=None):
        """
        Joint Variational Auto-Encoder using marginal and joint variational bounds,
        and constraining q(z|x,y) = q(z|x) q(z|y). Encoders and decoders are fully-
        connected networks with 2 hidden layers. 

        input_dim: tuple of output variable dimensions (assumes 2 output variables)
        latent_dim: latent space dimensionality
        learning_rate: optimization learning rate
        n_hidden_units: number of hidden units in each layer of encoder and decoder
        """
        # data and latent dimensionality
        self.x_dim, self.y_dim = input_dim
        self.z_dim = latent_dim
        self.lr = learning_rate
        self.n_hidden = n_hidden_units

        # counter for number of executed training steps
        self.n_steps = 0

        # base class constructor (initializes model)
        super(JointVAE, self).__init__(name=name, session=session, 
            log_dir=log_dir, model_dir=model_dir)


    def _initialize(self,):
        """
        Initialize model components. This method is invoked in object constructor. 
        """
        with tf.variable_scope(self.name, reuse=False):  
            
            # paired inputs X and Y
            self.X_joint = tf.placeholder(tf.float32, [None, self.x_dim], name='X_joint')
            self.Y_joint = tf.placeholder(tf.float32, [None, self.y_dim], name='Y_joint')

            # X input with missing Y
            self.X = tf.placeholder(tf.float32, [None, self.x_dim], name='X')

            # Y input with missing X
            self.Y = tf.placeholder(tf.float32, [None, self.y_dim], name='Y')

            # batch sizes for each case
            n_x = tf.shape(self.X)[0]
            n_y = tf.shape(self.Y)[0]
            n_xy = tf.shape(self.X_joint)[0]

            # q(z|x) and q(z|y) parameters
            self.zx_mean, self.zx_var = self._q_z_x(self.X, self.x_dim, self.z_dim, self.n_hidden,
                scope='q_z_x', reuse=False)
            self.zy_mean, self.zy_var = self._q_z_x(self.Y, self.y_dim, self.z_dim, self.n_hidden,
                scope='q_z_y', reuse=False)

            # q(z|x,y) parameters, defined using constraint q(z|x,y) = q(z|x) q(z|y)
            zxm, zxv = self._q_z_x(self.X_joint, self.x_dim, self.z_dim, self.n_hidden,
                scope='q_z_x', reuse=True)
            zym, zyv = self._q_z_x(self.Y_joint, self.y_dim, self.z_dim, self.n_hidden,
                scope='q_z_y', reuse=True)
            self.zxy_mean, self.zxy_var = self._constrain_joint(zxm, zxv, zym, zyv)

            # sampling 
            self.Zx = self._sample(self.zx_mean, self.zx_var, self.z_dim, n_x)
            self.Zy = self._sample(self.zy_mean, self.zy_var, self.z_dim, n_y)
            self.Zxy = self._sample(self.zxy_mean, self.zxy_var, self.z_dim, n_xy)

            # p(x|z) parameters
            self.xy_logits, self.xy_probs = self._p_x_z(self.Zy, self.z_dim, self.x_dim, self.n_hidden,
                scope='p_x_z', reuse=False)
            self.xx_logits, self.xx_probs = self._p_x_z(self.Zx, self.z_dim, self.x_dim, self.n_hidden,
                scope='p_x_z', reuse=True)
            
            # p(y|z) parameters
            self.yx_logits, self.yx_probs = self._p_x_z(self.Zx, self.z_dim, self.y_dim, self.n_hidden,
                scope='p_y_z', reuse=False)
            self.yy_logits, self.yy_probs = self._p_x_z(self.Zy, self.z_dim, self.y_dim, self.n_hidden,
                scope='p_y_z', reuse=True)
            
            # joint p(x|z) and p(y|z) parameters
            self.x_logits_joint, self.x_probs_joint = self._p_x_z(self.Zxy, self.z_dim, self.x_dim, 
                self.n_hidden, scope='p_x_z', reuse=True)
            self.y_logits_joint, self.y_probs_joint = self._p_x_z(self.Zxy, self.z_dim, self.y_dim, 
                self.n_hidden, scope='p_y_z', reuse=True)

            # bound: X observed, Y missing
            self.x_bound = self._marginal_bound(self.xx_logits, self.X, self.zx_mean, self.zx_var,
                scope='marginal_x')
            
            # bound: Y observed, X missing
            self.y_bound = self._marginal_bound(self.yy_logits, self.Y, self.zy_mean, self.zy_var,
                scope='marginal_y')

            # bound: X,Y observed
            self.xy_bound = self._joint_bound(self.x_logits_joint, self.X_joint, self.y_logits_joint, self.Y_joint, 
                self.zxy_mean, self.zxy_var, scope='joint_bound')

            # loss
            self.loss = -(self.x_bound + self.y_bound + self.xy_bound)

            # optimization step
            self.step = self._optimizer(self.loss, self.lr)


    def _q_z_x(self, X, x_dim, z_dim, n_hidden, scope, reuse):
        """
        Inference network.

        X: input data
        x_dim: dimensionality of input space 
        z_dim: dimensionality of latent space
        n_hidden: number of hidden units in each layer
        """
        with tf.variable_scope(scope, reuse=reuse):
            a1 = mod.affine_map(X, x_dim, n_hidden, "layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)
            a2 = mod.affine_map(h1, n_hidden, n_hidden, "layer_2", reuse=reuse)
            h2 = tf.nn.relu(a2)

            z_mean = mod.affine_map(h2, n_hidden, z_dim, "mean_layer", reuse=reuse)

            a3_var = mod.affine_map(h2, n_hidden, z_dim, "var_layer", reuse=reuse)
            z_var = tf.nn.softplus(a3_var)

            return z_mean, z_var


    def _constrain_joint(self, x_mean, x_var, y_mean, y_var):
        """
        Computes joint Gaussian as the product of two Gaussians with given means and variances. 
        """
        xv_inv = tf.reciprocal(x_var)
        yv_inv = tf.reciprocal(y_var)
        xy_var = tf.reciprocal(xv_inv + yv_inv)
        xx = tf.multiply(xv_inv, x_mean)
        yy = tf.multiply(yv_inv, y_mean)
        xy_mean = tf.multiply(xy_var, xx + yy)

        return xy_mean, xy_var


    def _p_x_z(self, Z, z_dim, x_dim, n_hidden, scope, reuse):
        """
        Generator network. 

        Z: latent data
        z_dim: dimensionality of latent space
        x_dim: dimensionality of output space
        n_hidden: number of hidden units in each layer
        """
        with tf.variable_scope(scope, reuse=reuse):
            a1 = mod.affine_map(Z, z_dim, n_hidden, "layer_1", reuse=reuse)
            h1 = tf.nn.relu(a1)
            a2 = mod.affine_map(h1, n_hidden, n_hidden, "layer_2", reuse=reuse)
            h2 = tf.nn.relu(a2)

            x_logits = mod.affine_map(h2, n_hidden, x_dim, "layer_3", reuse=reuse)
            x_probs = tf.nn.sigmoid(x_logits)

            return x_logits, x_probs


    def _sample(self, z_mean, z_var, z_dim, n_samples, scope='sampling', reuse=False):
        """
        Monte Carlo sampling from Gaussian distribution. Takes a single sample for each observation.
        """
        with tf.variable_scope(scope, reuse=reuse):
            z_std = tf.sqrt(z_var)
            eps = tf.random_normal((n_samples, z_dim))
            z = z_mean + tf.multiply(z_std, eps)
            return z
    

    def _marginal_bound(self, logits, labels, mean, var, scope='marginal_bound'):
        """
        Variational bound on marginal distribution p(x). 
        """
        with tf.variable_scope(scope, reuse=False):
            # reconstruction
            l1 = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, 
                labels=labels), axis=1)

            # penalty
            l2 = 0.5 * tf.reduce_sum(1 + tf.log(var) - tf.square(mean) - var, axis=1)

            # total bound
            return tf.reduce_mean(l1+l2, axis=0)
        

    def _joint_bound(self, x_logits, x_labels, y_logits, y_labels, mean, var, scope='joint_bound'):
        """
        Variational bound on joint distribution p(x,y).
        """
        with tf.variable_scope(scope, reuse=False):
            # reconstructions
            l1_x = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, 
                labels=x_labels), axis=1)
            l1_y = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logits, 
                labels=y_labels), axis=1)

            # penalty on q(z|x,y)
            l2 = 0.5 * tf.reduce_sum(1 + tf.log(var) - tf.square(mean) - var, axis=1)

            # total bound
            return tf.reduce_mean(l1_x + l1_y + l2, axis=0)


    def _optimizer(self, loss, learning_rate):
        """
        Optimization method.
        """
        with tf.variable_scope("optimization", reuse=False):
            step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
            return step


    def _summaries(self,):
        """
        Merge summary variables for visualizing with tensorboard.
        """
        with tf.variable_scope("summary", reuse=False):
            tf.summary.scalar('x_bound', self.x_bound)
            tf.summary.scalar('y_bound', self.y_bound)
            tf.summary.scalar('joint_bound', self.xy_bound)
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


    def translate_x(self, X):
        """
        Translate X to Y.
        """
        feed = {self.X: X}
        self.sess.run(self.yx_probs, feed_dict=feed)


    def translate_y(self, Y):
        """
        Translate Y to X.
        """
        feed = {self.Y: Y}
        self.sess.run(self.xy_probs, feed_dict=feed)


    def reconstruct(self, X_joint, Y_joint):
        """
        Reconstruct X and Y, given paired input X and Y.
        """
        feed = {self.X_joint: X_joint, self.Y_joint: Y_joint}
        return self.sess.run([self.x_probs_joint, self.y_probs_joint], feed_dict=feed)

    
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

