import tensorflow as tf
import numpy as np

from custom_GRU import (
    ln,
)

class fast_weights_model(object):

    def __init__(self, FLAGS):

        self.X = tf.placeholder(tf.float32,
            shape=[None, FLAGS.input_dim, FLAGS.num_classes], name='inputs_X')
        self.y = tf.placeholder(tf.float32,
            shape=[None, FLAGS.num_classes], name='targets_y')
        self.l = tf.placeholder(tf.float32, [], # need [] for tf.scalar_mul
            name="learning_rate")
        self.e = tf.placeholder(tf.float32, [],
            name="decay_rate")

        with tf.variable_scope("fast_weights"):

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

            # scale and shift for layernorm
            self.gain = tf.Variable(tf.ones(
                [FLAGS.num_hidden_units]),
                dtype=tf.float32)
            self.bias = tf.Variable(tf.zeros(
                [FLAGS.num_hidden_units]),
                dtype=tf.float32)

        # fast weights and hidden state initialization
        self.A = tf.zeros(
            [FLAGS.batch_size, FLAGS.num_hidden_units, FLAGS.num_hidden_units],
            dtype=tf.float32)
        self.h = tf.zeros(
            [FLAGS.batch_size, FLAGS.num_hidden_units],
            dtype=tf.float32)

        # NOTE:inputs are batch-major
        # Process batch by time-major
        for t in range(0, FLAGS.input_dim):

            # hidden state (preliminary vector)
            self.h = tf.nn.relu((tf.matmul(self.X[:, t, :], self.W_x)+self.b_x) +
                (tf.matmul(self.h, self.W_h)))

            # Forward weight and layer normalization
            if FLAGS.model_name == 'RNN-LN-FW':

                # Reshape h to use with a
                self.h_s = tf.reshape(self.h,
                    [FLAGS.batch_size, 1, FLAGS.num_hidden_units])

                # Create the fixed A for this time step
                self.A = tf.add(tf.scalar_mul(self.l, self.A),
                    tf.scalar_mul(self.e, tf.batch_matmul(tf.transpose(
                        self.h_s, [0, 2, 1]), self.h_s)))

                # Loop for S steps
                for _ in range(FLAGS.S):
                    self.h_s = tf.reshape(
                        tf.matmul(self.X[:, t, :], self.W_x)+self.b_x,
                        tf.shape(self.h_s)) + tf.reshape(
                        tf.matmul(self.h, self.W_h), tf.shape(self.h_s)) + \
                        tf.batch_matmul(self.h_s, self.A)

                    # Apply layernorm
                    mu = tf.reduce_mean(self.h_s, reduction_indices=0) # each sample
                    sigma = tf.sqrt(tf.reduce_mean(tf.square(self.h_s - mu),
                        reduction_indices=0))
                    self.h_s = tf.div(tf.mul(self.gain, (self.h_s - mu)), sigma) + \
                        self.bias

                    # Apply nonlinearity
                    self.h_s = tf.nn.relu(self.h_s)

                # Reshape h_s into h
                self.h = tf.reshape(self.h_s,
                    [FLAGS.batch_size, FLAGS.num_hidden_units])

            elif FLAGS.model_name == 'RNN-LN': # no fast weights but still LN

                # Apply layer norm
                with tf.variable_scope('just_ln') as scope:
                    if t > 0:
                        scope.reuse_variables()
                    self.h = ln(self.h, scope='h/')

            elif FLAGS.model_name == 'CONTROL': # no fast weights or LN
                pass

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

    def step(self, sess, batch_X, batch_y, l, e, forward_only=True):
        """
        Get results for training/validation.
        """
        input_feed = {self.X: batch_X, self.y: batch_y, self.l:l, self.e:e}

        if not forward_only: # training
            output_feed = [self.logits, self.loss, self.accuracy, self.norm,
            self.update]
        elif forward_only: # validation
            output_feed = [self.loss, self.accuracy]

        # process outputs
        outputs = sess.run(output_feed, input_feed)

        if not forward_only:
            return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
        elif forward_only:
            return outputs[0], outputs[1]







