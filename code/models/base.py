import tensorflow as tf
import os


class Model(object):

    def __init__(self, name, session=None, log_dir=None, model_dir=None):
        """
        Model base class to be used with tensorflow. Base class allows for 
        logging with tensorboard, as well as model load/save functionality. 

        name: model identifier 
        session: tensorflow session (optional)
        log_dir: directory for saving tensorboard log files
        model_dir: directory for saving model
        """
        self.name = name
        self.sess = session
        self.log_dir = log_dir
        self.model_dir = model_dir

        # create default directories
        if self.log_dir is None:
            self.log_dir = '../logs/'
        if self.model_dir is None: 
            self.model_dir = '../models/'

        # create directories if they don't exist
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        # model initialization
        self._initialize()

        # summary variables 
        self.summary = self._summaries()

        # summary writers
        self.tr_writer = tf.summary.FileWriter(self.log_dir+self.name+'_train') 
        self.te_writer = tf.summary.FileWriter(self.log_dir+self.name+'_test') 

        # tensorflow session
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        # visualize tf graph
        self.tr_writer.add_graph(self.sess.graph)
        self.te_writer.add_graph(self.sess.graph)
      
    
    def _initialize(self,):
        """
        Model initialization code to be placed here. 
        """
        pass


    def _summaries(self,):
        """
        Summary variables for visualizing with tensorboard.
        """
        pass


    def save_state(self, name=None, suffix=None):
        """
        Save model.
        """
        if name is None:
            name = self.name
        if suffix is not None:
            name = name + '_' + suffix
        saver = tf.train.Saver()
        saver.save(self.sess, self.model_dir+name)


    def load_state(self, name=None, suffix=None):
        """
        Load model.
        """
        if name is None:
            name = self.name
        if suffix is not None:
            name = name + '_' + suffix
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_dir+name)


    def close(self, ):
        """
        Close session (Need to close session before loading another model)
        """
        self.sess.close()
        tf.reset_default_graph()






