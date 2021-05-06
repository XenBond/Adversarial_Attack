
"""
A pure TensorFlow implementation of a convolutional neural network.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import tensorflow as tf

from cleverhans import initializers
from cleverhans.model import Model
from mislnet_model import MISLNet


class MISLNet_Features(Model):
    def __init__(self, scope):
        Model.__init__(self, scope, 200, locals())
        # self.nb_filters = nb_filters

        # Do a dummy run of fprop to make sure the variables are created from
        # the start
        self.fprop(tf.placeholder(tf.float32, [None, 256, 256, 3]))
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x):
        print('init')
        logits=MISLNet(x, False, reuse=tf.AUTO_REUSE, namescope=self.scope)
        print('finish')
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            return {self.O_LOGITS: logits,
                    self.O_PROBS: logits}
