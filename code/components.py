import tensorflow as tf
import numpy as np
import math
import os


class VariationalAutoEncoder(object):

    def __init__(self, input_dim, latent_dim, learning_rate, model_name, session=None,
        epsilon = 1e-3, decay = 0.99, 
        log_dir='../logs/', model_dir='../models/'):
        """
        Variational Auto-Encoder. 

        input_dim: input data dimensions
        latent_dim: latent space dimensionality
        learning_rate: optimization learning rate
        model_name: identifier for current experiment
        epsilon/decay: batch normalization parameters
        """
        # data and latent dimensionality
        self.x_dim = input_dim
        self.z_dim = latent_dim
        self.lr = learning_rate
        self.name = model_name
        self.sess = session
        self.eps = epsilon
        self.decay = decay

        with tf.variable_scope(self.name, reuse=False):  
            # training indicator
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            # input placeholder
            self.X = tf.placeholder(tf.float32, [None, self.x_dim], name='X')

            # latent space parameters
            self.z_mean, self.z_var = self._encoder()

            # samples from latent space
            self.Z = self._sample_latent_space()

            # model parameters
            self.x_logits, self.x_probs = self._decoder()      

            # variational bound
            self.bound = self._variational_bound()

            # loss
            self.loss = -self.bound  

            # optimization step
            self.step = self._optimizer()

            # summary variables 
            self.summary = self._summaries()

            # logger and model directories
            self.log_dir = log_dir
            self.model_dir = model_dir
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

            # summary writers
            self.tr_writer = tf.summary.FileWriter(self.log_dir+self.name+'_train') 
            self.te_writer = tf.summary.FileWriter(self.log_dir+self.name+'_test') 

        # counts number of executed training steps
        self.n_steps = 0

        # tensorflow session
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
          

    def _encoder(self,):
        """
        Recognition network.
        """
        with tf.variable_scope("encoder", reuse=False):
            a1 = affine_map(self.X, self.x_dim, 500, "layer_1")
            b1 = batch_norm(a1, 'b1', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h1 = tf.nn.relu(b1)
            a2 = affine_map(h1, 500, 500, "layer_2")
            b2 = batch_norm(a2, 'b2', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h2 = tf.nn.relu(b2)

            a3_mean = affine_map(h2, 500, self.z_dim, "mean_layer")
            z_mean = batch_norm(a3_mean, 'z_mean', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False) 

            a3_var = affine_map(h2, 500, self.z_dim, "var_layer")
            b3_var = batch_norm(a3_var, 'b3_var', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False) 
            z_var = tf.nn.softplus(b3_var)

            return z_mean, z_var


    def _decoder(self,):
        """
        Generator network. 
        """
        with tf.variable_scope("decoder", reuse=False):
            a1 = affine_map(self.Z, self.z_dim, 500, "layer_1")
            b1 = batch_norm(a1, 'b1', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h1 = tf.nn.relu(b1)
            a2 = affine_map(h1, 500, 500, "layer_2")
            b2 = batch_norm(a2, 'b2', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h2 = tf.nn.relu(b2)

            a3 = affine_map(h2, 500, self.x_dim, "layer_3")
            x_logits = batch_norm(a3, 'x_logits', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            x_probs = tf.nn.sigmoid(x_logits)

            return x_logits, x_probs

    
    def _sample_latent_space(self,):
        """
        Monte Carlo sampling from Latent distribution. Takes a single sample for each observation.
        """
        with tf.variable_scope("sampling", reuse=False):
            z_std = tf.sqrt(self.z_var)
            mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=self.z_mean, scale_diag=z_std)
            return tf.squeeze(mvn.sample())
    

    def _variational_bound(self,):
        """
        Variational Bound.
        """
        with tf.variable_scope("variational_bound", reuse=False):
            # reconstruction
            l1 = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_logits, 
                labels=self.X), axis=1)

            # penalty
            l2 = 0.5 * tf.reduce_sum(1 + tf.log(self.z_var) - tf.square(self.z_mean) - self.z_var, axis=1)

            # total bound
            return tf.reduce_mean(l1+l2, axis=0)
        

    def _optimizer(self,):
        """
        Optimization method.
        """
        with tf.variable_scope("optimization", reuse=False):
            step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            return step


    def _summaries(self,):
        """
        Summary variables for visualizing with tensorboard.
        """
        with tf.variable_scope("summary", reuse=False):
            tf.summary.scalar('variational_bound', self.bound)
            #tf.summary.histogram('z_mean', self.z_mean)
            #tf.summary.histogram('z_var', self.z_var)
            return tf.summary.merge_all()


    def train(self, batch_X, write=True):
        """
        Executes single training step.

        batch_X: minibatch of input data
        write: indicates whether to write summary
        """
        feed = {self.X: batch_X, self.is_training: True}
        summary, _ = self.sess.run([self.summary, self.step], feed_dict=feed)
        if write:
            self.tr_writer.add_summary(summary, self.n_steps)
        self.n_steps = self.n_steps + 1
    

    def test(self, batch_X):
        """
        Writes summary for test data.

        batch_X: minibatch of input data
        """
        feed = {self.X: batch_X, self.is_training: False}
        loss, summary = self.sess.run([self.loss, self.summary], feed_dict=feed)
        self.te_writer.add_summary(summary, self.n_steps)

        return loss
        

    def reconstruct(self, batch_X):
        """
        Reconstructed data, given input X.

        batch_X: minibatch of input data
        """
        feed = {self.X: batch_X, self.is_training: False}
        return self.sess.run(self.x_probs, feed_dict=feed)

    
    def encode(self, batch_X):
        """
        Computes mean of latent space, given input X.

        batch_X: minibatch of input data        
        """
        feed = {self.X: batch_X, self.is_training: False}
        return self.sess.run(self.z_mean, feed_dict=feed)

    
    def decode(self, batch_Z):
        """
        Computes bernoulli probabilities in data space, given input Z.

        batch_Z: minibatch in latent space
        """
        feed = {self.Z: batch_Z, self.is_training: False}
        return self.sess.run(self.x_probs, feed_dict=feed)


    def save_state(self, name=None):
        """
        Save model.
        """
        if name is None:
            name = self.name
        saver = tf.train.Saver()
        saver.save(self.sess, self.model_dir+name)


    def load_state(self, name=None):
        """
        Load model.
        """
        if name is None:
            name = self.name
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_dir+name)




def affine_map(input, in_dim, out_dim, scope, reuse=False):
    """
    Affine transform.

    input: input tensor
    in_dim/out_dim: input and output dimensions
    scope: variable scope as string
    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable("W", shape=[in_dim,out_dim], 
            initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
        b = tf.get_variable("b", shape=[out_dim], initializer=tf.constant_initializer(0))

        return tf.matmul(input,W) + b


def batch_norm(x, scope, decay, epsilon, is_training, center=True, reuse=False):
    """
    Batch normalization layer

      This was implemented while referring to the following papers/resources:
      https://arxiv.org/pdf/1502.03167.pdf
      https://arxiv.org/pdf/1603.09025.pdf
      http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    """
    with tf.variable_scope(scope, reuse=reuse):
        # number of features in input matrix
        dim = x.get_shape()[1].value
        # scaling coefficients
        gamma = tf.get_variable('gamma', [dim], initializer=tf.constant_initializer(1.0), trainable=True)
        # offset coefficients (if required)
        if center:
            beta = tf.get_variable('beta', [dim], initializer=tf.constant_initializer(0), trainable=True)
        else:
            beta = None

        # population mean variable (for prediction)
        popmean = tf.get_variable('pop_mean', shape=[dim], initializer=tf.constant_initializer(0.0),
            trainable=False)
        # population variance variable (for prediction)
        popvar = tf.get_variable('pop_var', shape=[dim], initializer=tf.constant_initializer(1.0), 
            trainable=False)

        # compute batch mean and variance
        batch_mean, batch_var = tf.nn.moments(x, axes=[0])

        def update_and_train():
            # update popmean and popvar using moving average of batch_mean and batch_var
            pop_mean_new = popmean * decay + batch_mean * (1 - decay)
            pop_var_new = popvar * decay + batch_var * (1 - decay)
            with tf.control_dependencies([popmean.assign(pop_mean_new), popvar.assign(pop_var_new)]):
                # batch normalization
                return tf.nn.batch_normalization(x, mean=batch_mean, variance=batch_var, 
                    offset=beta, scale=gamma, variance_epsilon=epsilon)

        def predict():
            # batch normalization (using population moments)
            return tf.nn.batch_normalization(x, mean=popmean, variance=popvar, 
                offset=beta, scale=gamma, variance_epsilon=epsilon)

        # conditional evaluation in tf graph
        return tf.cond(is_training, update_and_train, predict)



class M2(object):

    def __init__(self, input_dim, latent_dim, n_classes, learning_rate, model_name, session=None,
        epsilon = 1e-3, decay = 0.99,  
        log_dir='../logs/', model_dir='../models/'):
        """
        This is the M2 model based on this paper: https://arxiv.org/abs/1406.5298
        M2 input is treated as Gaussian, as is the case in the M1+M2 configuration. 
        
        input_dim: input data dimensions
        latent_dim: latent space dimensionality
        y_dim: number of classes (y is a one-hot encoded vector)
        learning_rate: optimization learning rate
        model_name: identifier for current experiment
        epsilon/decay: batch normalization parameters
        """
        self.x_dim = input_dim
        self.z_dim = latent_dim
        self.y_dim = n_classes
        self.lr = learning_rate
        self.name = model_name
        self.sess = session
        self.eps = epsilon
        self.decay = decay

        with tf.variable_scope(self.name, reuse=False):  
            # training indicator
            self.is_training = tf.placeholder(tf.bool, name='is_training')  

            # labelled input placeholders
            self.X = tf.placeholder(tf.float32, [None, self.x_dim], name='X')
            self.Y = tf.placeholder(tf.float32, [None, self.y_dim],name='Y')

            # missing and labelled index
            self.missing = tf.placeholder(tf.int32, [None], name='missing_index')
            self.labelled = tf.placeholder(tf.int32, [None], name='labelled_index')

            # predictive y probabilities using q(y|x) 
            self.Y_logits, self.Y_prob = self._discriminator(self.X)

            # separate missing and labelled probabilities
            self.Ymiss_logits = tf.gather(params=self.Y_logits, indices=self.missing)
            self.Ymiss_prob = tf.gather(params=self.Y_prob, indices=self.missing)
            self.Ylab_logits = tf.gather(params=self.Y_logits, indices=self.labelled)
            self.Ylab_prob = tf.gather(params=self.Y_prob, indices=self.labelled)

            # sample missing labels 
            self.Ymiss = self._sample_y(self.Ymiss_prob, scope='missing_y_sampled') 

            # separate missing and labelled data
            self.Xmiss = tf.gather(params=self.X, indices=self.missing)
            self.Ylab = tf.gather(params=self.Y, indices=self.labelled)
            self.Xlab = tf.gather(params=self.X, indices=self.labelled)

            # update input tensors
            self.XX = tf.concat([self.Xlab, self.Xmiss], axis=0)
            self.YY = tf.concat([self.Ylab, self.Ymiss], axis=0)

            # encoder
            self.z_mean, self.z_var = self._encoder(self.XX, self.YY)
        
            # sample z 
            self.Z = self._sample_z(self.z_mean, self.z_var)

            # decoder 
            self.x_mean, self.x_var = self._decoder(self.YY, self.Z)        
            
            # variational bound 
            self.bound = self._variational_bound(self.XX, self.YY, self.x_mean, self.x_var, 
                self.z_mean, self.z_var, self.Ymiss_logits, self.Ymiss_prob)

            # classification loss (negative cross entropy)
            self.class_loss = self._classification_loss(self.Ylab, self.Ylab_logits)

            # classification accuracy
            self.accuracy = self._accuracy(self.Ylab, self.Ylab_logits)

            # optimization objective
            self.loss = -self.bound + self.class_loss

            # optimization step
            self.step = self._optimizer()

            # summary variables 
            self.summary = self._summaries()

            # logger and model directories
            self.log_dir = log_dir
            self.model_dir = model_dir
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

            # summary writers
            self.tr_writer = tf.summary.FileWriter(self.log_dir+self.name+'_train') 
            self.te_writer = tf.summary.FileWriter(self.log_dir+self.name+'_test') 

        # counts number of executed training steps
        self.n_steps = 0

        # tensorflow session
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
          

    def _discriminator(self, X, scope="classifier", reuse=False):
        """
        Discriminative classifier q(y|x).
        """
        with tf.variable_scope(scope, reuse=reuse):  
            n_width = (self.x_dim + self.y_dim) / 2

            a1 = affine_map(X, self.x_dim, n_width, "layer_1", reuse=reuse)
            b1 = batch_norm(a1, 'b1', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False, reuse=reuse)
            h1 = tf.nn.relu(b1)
            a2 = affine_map(h1, n_width, n_width, "layer_2", reuse=reuse)
            b2 = batch_norm(a2, 'b2', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False, reuse=reuse)
            h2 = tf.nn.relu(b2)

            a3 = affine_map(h2, n_width, self.y_dim, "layer_3", reuse=reuse)
            y_logits = batch_norm(a3, 'y_logits', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False, reuse=reuse)
            y_probs = tf.nn.softmax(y_logits)

            return y_logits, y_probs


    def _encoder(self, X, Y, scope="encoder"):
        """
        Recognition network.
        """
        with tf.variable_scope(scope, reuse=False):
            input = tf.concat([X,Y], axis=1)

            n_width = (self.x_dim + self.y_dim + self.z_dim) / 2

            a1 = affine_map(input, self.x_dim+self.y_dim, n_width, "layer_1")
            b1 = batch_norm(a1, 'b1', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h1 = tf.nn.relu(b1)
            a2 = affine_map(h1, n_width, n_width, "layer_2")
            b2 = batch_norm(a2, 'b2', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h2 = tf.nn.relu(b2)

            a3_mean = affine_map(h2, n_width, self.z_dim, "mean_layer")
            z_mean = batch_norm(a3_mean, 'z_mean', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False) 

            a3_var = affine_map(h2, n_width, self.z_dim, "var_layer")
            b3_var = batch_norm(a3_var, 'b3_var', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False) 
            z_var = tf.nn.softplus(b3_var)

            return z_mean, z_var


    def _decoder(self, Y, Z, scope="decoder"):
        """
        Generator network. 
        """
        with tf.variable_scope(scope, reuse=False):
            input = tf.concat([Y,Z], axis=1)

            n_width = (self.y_dim + self.z_dim + self.x_dim) / 2

            a1 = affine_map(input, self.y_dim + self.z_dim, n_width, "layer_1")
            b1 = batch_norm(a1, 'b1', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h1 = tf.nn.relu(b1)
            a2 = affine_map(h1, n_width, n_width, "layer_2")
            b2 = batch_norm(a2, 'b2', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False)
            h2 = tf.nn.relu(b2)

            a3_mean = affine_map(h2, n_width, self.x_dim, "mean_layer")
            x_mean = batch_norm(a3_mean, 'x_mean', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False) 

            a3_var = affine_map(h2, n_width, self.x_dim, "var_layer")
            b3_var = batch_norm(a3_var, 'b3_var', decay=self.decay, epsilon=self.eps,
                is_training=self.is_training, center=False) 
            x_var = tf.nn.softplus(b3_var)

            return x_mean, x_var


    def _sample_y(self, y_probs, scope='y_sampler'):
        """
        Monte Carlo sampling from q(y|x). Returns single sample for each observation.
        """
        with tf.variable_scope(scope, reuse=False):
            cat = tf.contrib.distributions.OneHotCategorical(probs=y_probs)
            return tf.cast(tf.squeeze(cat.sample()), tf.float32)

    
    def _sample_z(self, z_mean, z_var, scope='z_sampler'):
        """
        Monte Carlo sampling from q(z|x,y). Returns a single sample for each observation.
        """
        with tf.variable_scope(scope, reuse=False):
            z_std = tf.sqrt(z_var)
            mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=z_std)
            return tf.squeeze(mvn.sample())
    

    def _variational_bound(self, X, Y, x_mean, x_var, z_mean, z_var, 
        Ymiss_logits, Ymiss_prob, scope='variational_bound'):
        """
        Variational Bound.
        """
        with tf.variable_scope(scope, reuse=False):

            # reconstruction 
            l1a = self.x_dim * math.log(2*math.pi)
            l1b = tf.reduce_sum(tf.log(x_var), axis=1)
            l1c = tf.reduce_sum(tf.multiply(tf.square(X - x_mean), x_var), axis=1)
            l1 = -0.5 * (l1a + l1b + l1c) 

            # penalty
            l2 = 0.5 * tf.reduce_sum(1 + tf.log(self.z_var) - tf.square(self.z_mean) - self.z_var, axis=1)

            # log p(y) --> assuming uniform prior on y (using labelled y only)
            logpy = math.log(0.1)            

            # entropy of q(y|x) for missing data 
            entropy = tf.reduce_sum(-tf.multiply(Ymiss_prob, tf.nn.log_softmax(Ymiss_logits)), axis=1)

            # negative cross entropy of p(y)=0.1 on q(y|x)
            xent = tf.reduce_sum(math.log(0.1) * Ymiss_prob, axis=1)

            # bound components
            missing_bound = tf.reduce_mean(entropy + xent, axis=0)
            labelled_bound = logpy 
            combined_bound = tf.reduce_mean(l1 + l2, axis=0)

            # total bound
            return missing_bound + labelled_bound + combined_bound
        

    def _classification_loss(self, Ylab, Ylab_logits, scope='classification_loss'):
        """
        Classification loss --> based on symmetric Dirichlet prior on parameters of p(y)
        """
        with tf.variable_scope(scope, reuse=False):
            alpha = 0.1 
            xent_lab = -tf.reduce_sum(tf.multiply(Ylab, tf.nn.log_softmax(Ylab_logits)), axis=1)
            return alpha * tf.reduce_mean(xent_lab, axis=0)

    
    def _accuracy(self, Y, Y_logits, scope='accuracy'):
        """
        Classification accuracy using argmax q(y|x)
        """
        with tf.variable_scope(scope, reuse=False):
            pred = tf.argmax(Y_logits, axis=1)
            truth = tf.argmax(Y, axis=1)
            correct = tf.equal(truth, pred)
            return tf.reduce_mean(tf.cast(correct, tf.float32))
            

    def _optimizer(self,):
        """
        Optimization method
        """
        with tf.variable_scope("optimization", reuse=False):
            step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            return step


    def _summaries(self,):
        """
        Summary variables for visualizing with tensorboard.
        """
        with tf.variable_scope("summary", reuse=False):
            tf.summary.scalar('variational_bound', self.bound)
            tf.summary.scalar('classification_loss', self.class_loss)
            tf.summary.scalar('classification_accuracy', self.accuracy)
            tf.summary.scalar('total_loss', self.loss)
            #tf.summary.histogram('z_mean', self.z_mean)
            #tf.summary.histogram('z_var', self.z_var)
            return tf.summary.merge_all()


    def train(self, X, Y, missing, labelled, write=True):
        """
        Executes single training step.

        X/Y: minibatch of examples
        missing: indicates missing examples
        labelled: indicates labelled examples
        write: indicates whether to write summary
        """
        feed = {self.X: X, self.Y: Y, self.missing: missing, self.labelled: labelled, self.is_training: True}
        summary, _ = self.sess.run([self.summary, self.step], feed_dict=feed)
        if write:
            self.tr_writer.add_summary(summary, self.n_steps)
        self.n_steps = self.n_steps + 1
    

    def test(self, X, Y):
        """
        Writes summary for test data. Returns classification accuracy. 
        """
        missing = np.array([])
        labelled = np.arange(X.shape[0])
        feed = {self.X: X, self.Y: Y, self.missing: missing, self.labelled: labelled, self.is_training: False}
        accuracy, summary = self.sess.run([self.accuracy, self.summary], feed_dict=feed)
        self.te_writer.add_summary(summary, self.n_steps)

        return accuracy
        

    def predict(self, X):
        """
        Computes class probabilities using q(y|x) distribution. 
        """
        feed = {self.X: X, self.is_training: False}
        return self.sess.run(self.Y_prob, feed_dict=feed)


    def save_state(self, name=None):
        """
        Save model.
        """
        if name is None:
            name = self.name
        saver = tf.train.Saver()
        saver.save(self.sess, self.model_dir+name)


    def load_state(self, name=None):
        """
        Load model.
        """
        if name is None:
            name = self.name
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_dir+name)



