import tensorflow as tf
import numpy as np

import collections
import math

from tensorflow.python.framework import (
    ops,
    tensor_shape,
)

from tensorflow.python.ops import (
    array_ops,
    clip_ops,
    embedding_ops,
    init_ops,
    math_ops,
    nn_ops,
    partitioned_variables,
    variable_scope as vs,
)

from tensorflow.python.ops.math_ops import (
    sigmoid,
    tanh,
)

from tensorflow.python.platform import (
    tf_logging as logging,
)

from tensorflow.python.util import (
    nest,
)

# LN funcition
def ln(inputs, epsilon=1e-5, scope=None):

    """ Computer LN given an input tensor. We get in an input of shape
    [N X D] and with LN we compute the mean and var for each individual
    training point across all it's hidden dimensions rather than across
    the training batch as we do in BN. This gives us a mean and var of shape
    [N X 1].
    """
    mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
    with tf.variable_scope(scope + 'LN'):
        scale = tf.get_variable('alpha',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('beta',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift

    return LN

# Modified from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py
class GRUCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
          logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
          with vs.variable_scope("Gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            r, u = array_ops.split(1, 2, tf.nn.rnn_cell._linear([inputs, state],
                           2 * self._num_units, True, 1.0))

            # Apply Layer Normalization to the two gates
            r = ln(r, scope = 'r/')
            u = ln(r, scope = 'u/')

            r, u = sigmoid(r), sigmoid(u)
          with vs.variable_scope("Candidate"):
            c = self._activation(tf.nn.rnn_cell._linear([inputs, r * state],
                                         self._num_units, True))
          new_h = u * state + (1 - u) * c
        return new_h, new_h


class gru_model(object):

    def __init__(self, FLAGS):

        self.X = tf.placeholder(tf.float32,
            shape=[None, FLAGS.input_dim, FLAGS.num_classes], name='inputs_X')
        self.y = tf.placeholder(tf.float32,
            shape=[None, FLAGS.num_classes], name='targets_y')
        self.l = tf.placeholder(tf.float32, [], # need [] for tf.scalar_mul
            name="learning_rate")
        self.e = tf.placeholder(tf.float32, [],
            name="decay_rate")

        with tf.variable_scope("GRU"):

            # input weights (proper initialization)
            self.W_x = tf.Variable(tf.random_uniform(
                [FLAGS.num_classes, FLAGS.num_hidden_units],
                -np.sqrt(2.0/FLAGS.num_classes),
                np.sqrt(2.0/FLAGS.num_classes)),
                dtype=tf.float32)
            self.b_x = tf.Variable(tf.zeros(
                [FLAGS.num_hidden_units]),
                dtype=tf.float32)

            # hidden weights (See Hinton's video @ 21:20)
            self.W_h = tf.Variable(
                initial_value=0.05 * np.identity(FLAGS.num_hidden_units),
                dtype=tf.float32)

            # softmax weights (proper initialization)
            self.W_softmax = tf.Variable(tf.random_uniform(
                [FLAGS.num_hidden_units, FLAGS.num_classes],
                -np.sqrt(2.0 / FLAGS.num_hidden_units),
                np.sqrt(2.0 / FLAGS.num_hidden_units)),
                dtype=tf.float32)
            self.b_softmax = tf.Variable(tf.zeros(
                [FLAGS.num_classes]),
                dtype=tf.float32)

        self.h = tf.zeros(
            [FLAGS.batch_size, FLAGS.num_hidden_units],
            dtype=tf.float32)

        # GRU
        self.gru = GRUCell(FLAGS.num_hidden_units)
        with tf.variable_scope("gru_step") as scope:
            for t in range(0, FLAGS.input_dim):
                if t > 0:
                    scope.reuse_variables()
                self.outputs, self.h = self.gru(self.X[:, t, :], self.h)

        # All inputs processed! Time for softmax
        self.logits = tf.matmul(self.h, self.W_softmax) + self.b_softmax

        # Loss
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y))

        # Optimization
        self.lr = tf.Variable(0.0, trainable=False)
        self.trainable_vars = tf.trainable_variables()
        # clip the gradient to avoid vanishing or blowing up gradients
        self.grads, self.norm = tf.clip_by_global_norm(
            tf.gradients(self.loss, self.trainable_vars), FLAGS.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.update = optimizer.apply_gradients(
            zip(self.grads, self.trainable_vars))

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1),
            tf.argmax(self.y, 1)), tf.float32))

        # Components for model saving
        self.global_step = tf.Variable(0, trainable=False) # won't step
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, sess, batch_X, batch_y, l, e, forward_only):
        """
        Get results for training/validation.
        """
        input_feed = {self.X: batch_X, self.y: batch_y, self.l:l, self.e:e}

        if not forward_only: # training
            output_feed = [self.loss, self.accuracy, self.norm,
            self.update]
        elif forward_only: # validation
            output_feed = [self.loss, self.accuracy]

        # process outputs
        outputs = sess.run(output_feed, input_feed)

        if not forward_only:
            return outputs[0], outputs[1], outputs[2], outputs[3]
        elif forward_only:
            return outputs[0], outputs[1]







