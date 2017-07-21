import tensorflow as tf
import numpy as np

from models import base
from models import layers as nw


class PixelCNN(base.Model):
    """
    PixelCNN model

    Arguments:
    n_channels: number of channels in input images
    n_layers: number of residual blocks in pixel cnn
    learning_rate: optimizer learning_rate
    """
    def __init__(self, arguments, name, tracker, session=None, log_dir=None, model_dir=None):
        # dictionary of model/inference arguments
        self.args = arguments

        # object to track model performance (can be None)
        self.tracker = tracker

        # training steps counter
        self.n_steps = 0

        # base class constructor (initializes model)
        super(PixelCNN, self).__init__(name=name, session=session, log_dir=log_dir, model_dir=model_dir)


    def _initialize(self, ):
        # input dimensions
        n_layers = self.args['n_pixelcnn_layers']
        n_fmaps = self.args['n_feature_maps']
        self.n_ch = self.args['n_channels']
        self.h = self.args['height']
        self.w = self.args['width']
        self.n_x = self.h * self.w * self.n_ch

        # input placeholders
        self.x = tf.placeholder(tf.float32, [None, self.n_x], name='x')

        # shape to image
        x = tf.reshape(self.x, shape=[-1, self.h, self.w, self.n_ch])

        # pixel cnn model
        x = nw.pixel_cnn(x, n_layers, ka=7, kb=3, out_ch=self.n_ch, n_feature_maps=n_fmaps,
                         scope='pixel_cnn', reuse=False)

        # flatten
        logits = tf.reshape(x, shape=[-1, self.n_x])

        # autoregressive probabilities
        self.probs = self._probs(logits)

        # loss function
        self.loss = self._loss(labels=self.x, logits=logits)

        # optimizer
        self.step = self._optimizer(self.loss)


    def _probs(self, logits):
        return tf.nn.sigmoid(logits)


    def _loss(self, labels, logits):
        # maximize likelihood with autoregressive model
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_sum(loss, axis=1)
        loss = tf.reduce_mean(loss, axis=0)

        return loss


    def _optimizer(self, loss, scope='optimizer', reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            lr = self.args['learning_rate']
            step = tf.train.RMSPropOptimizer(lr).minimize(loss)

            return step


    def _summaries(self,):

        with tf.variable_scope("summaries", reuse=False):
            tf.summary.scalar('loss', self.loss)

            return tf.summary.merge_all()


    def _track(self, terms, prefix):

        if self.tracker is not None:

            for name, term in terms.items():
                self.tracker.add(i=self.n_steps, value=term, series_name=prefix + name, run_name=self.name)


    def train(self, x):
        """
        Performs single training step.
        """
        feed = {self.x: x}
        outputs = [self.summary, self.step, self.loss]

        summary, _, loss = self.sess.run(outputs, feed_dict=feed)

        # track performance
        terms = {'loss': loss}
        self._track(terms, prefix='train_')
        self.tr_writer.add_summary(summary, self.n_steps)

        self.n_steps = self.n_steps + 1


    def test(self, x):
        """
        Test loss
        """
        feed = {self.x: x}
        outputs = [self.summary, self.loss]

        summary, loss  = self.sess.run(outputs, feed_dict=feed)

        # track performance
        terms = {'loss': loss}
        self._track(terms, prefix='test_')
        self.te_writer.add_summary(summary, self.n_steps)


    def reconstruct(self, x):
        """
        Synthesize images autoregressively.
        """
        n_pixels = self.args['n_conditional_pixels']

        def _locate_2d(idx, w):
            pos = idx + 1
            r = np.ceil(pos / w)
            c = pos - (r-1)*w

            return int(r-1), int(c-1)

        h = self.h
        w = self.w
        ch = self.n_ch
        n_x = h * w * ch

        x = x.copy()

        remain = h*w - n_pixels

        for i in range(remain):
            feed = {self.x: x}
            probs = self.sess.run(self.probs, feed_dict=feed)
            probs = np.reshape(probs, newshape=[-1, h, w, ch])

            hp, wp = _locate_2d(n_pixels + i, w)

            x = np.reshape(x, newshape=[-1, h, w, ch])
            x[:, hp, wp, :] = np.random.binomial(n=1, p=probs[:, hp, wp, :])
            x = np.reshape(x, newshape=[-1, n_x])

        return x
