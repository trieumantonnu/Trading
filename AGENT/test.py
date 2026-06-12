import os
import numpy as np
import tensorflow as tf

# class DeepQNetwork(object):
#     def __init__(self, lr, n_actions, name, fcl_dims=256,
#                  input_dims=(210,160, 4), chkpt_dir='tmp/dqn'):
#         self.lr = lr
#         self.name = name
#         self.n_actions = n_actions
#         self.fcl_dims = fcl_dims
#         self.input_dims = input_dims
#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_intializer())
#         self.saver = tf.train.Saver()
#         self.checkpoint_file = os.path.join(chkpt_dir, 'deepqnet.ckpt')
#         self.params = tf.get_collection(tf.GraphKeys)

tf.compat.v1.disable_eager_execution() 

class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, fcl_dims=256,
                 input_dims=(210, 160, 4), chkpt_dir='tmp/dqn'):
        self.lr = lr
        self.name = name
        self.n_actions = n_actions
        self.fcl_dims = fcl_dims
        self.input_dims = input_dims

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.saver = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, 'deepqnet.ckpt')

        self.params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.name
        )