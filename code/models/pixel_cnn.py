import tensorflow as tf
import numpy as np

from models import base
from models import networks as nw


class PixelCNN(base.Model):
    """
    PixelCNN model

    Arguments:
    n_channels: number of channels in input images
    n_layers: number of residual blocks in pixel cnn
    learning_rate: optimizer learning_rate
    """
    def __init__(self, arguments, name="PixelCNN", session=None, log_dir=None, model_dir=None):
        # dictionary of model/inference arguments
        self.args = arguments

        # training steps counter
        self.n_steps = 0

        # base class constructor (initializes model)
        super(PixelCNN, self).__init__(name=name, session=session, log_dir=log_dir, model_dir=model_dir)


    def _initialize(self, ):
        # input dimensions
        self.n_ch = self.args['n_channels']
        self.h = 28
        self.w = 28
        self.n_x = self.h * self.w * self.n_ch
        n_layers = self.args['n_layers']

        # input placeholders
        self.x = tf.placeholder(tf.float32, [None, self.n_x], name='x')

        # shape to image
        x = tf.reshape(self.x, shape=[-1, self.h, self.w, self.n_ch])

        # pixel cnn model
        x = nw.pixel_cnn(x, n_layers, k=3, out_ch=self.n_ch, scope='pixel_cnn', reuse=False)

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


    def train(self, x, write=True):
        """
        Performs single training step.
        """
        feed = {self.x: x}
        outputs = [self.summary, self.step, self.loss]

        summary, _, loss = self.sess.run(outputs, feed_dict=feed)
        if write:
            self.tr_writer.add_summary(summary, self.n_steps)
        self.n_steps = self.n_steps + 1

        return loss


    def test(self, x):
        """
        Test loss
        """
        feed = {self.x: x}
        outputs = [self.summary, self.loss]

        summary, loss  = self.sess.run(outputs, feed_dict=feed)
        self.te_writer.add_summary(summary, self.n_steps)

        return loss


    def predict(self, x, n_pixels):
        """
        Synthesize images.

        n_pixels: number of pixels to condition on (in row-wise order)
        """
        def _locate_2d(idx, w):
            pos = idx + 1
            r = np.ceil(pos / w)
            c = pos - (r-1)*w

            return int(r-1), int(c-1)

        h = self.h
        w = self.w
        ch = self.n_ch
        n_x = h * w * ch

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
