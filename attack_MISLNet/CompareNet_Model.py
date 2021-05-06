
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
from cam_128.mislnet_model import CompareNet_v1_128


class CompareNet_Model(Model):
    def __init__(self, scope, batch_size, reference):
        Model.__init__(self, scope, 2, locals())
        # Do a dummy run of fprop to make sure the variables are created from
        # the start
        self.batch_size = batch_size
        self.reference = tf.placeholder_with_default(tf.constant(reference, dtype=tf.float32), shape=[batch_size, 128, 128, 3])
        self.fprop(tf.placeholder(tf.float32, [batch_size, 128, 128, 3]))
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x):
        print('init')
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            logits=CompareNet_v1_128(x, self.reference, False)
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}
