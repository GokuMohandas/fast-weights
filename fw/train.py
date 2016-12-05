import cPickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import sys

from data_utils import (
    generate_epoch,
)

from model import (
    fast_weights_model,
)

from custom_GRU import (
    gru_model,
)

class parameters():

    def __init__(self):

        self.input_dim = 9
        self.num_classes = 26+10+1
        self.num_epochs = 1000
        self.batch_size = 128

        self.num_hidden_units = 50
        self.l = 0.95 # decay lambda
        self.e = 0.5 # learning rate eta
        self.S = 1 # num steps to get to h_S(t+1)
        self.learning_rate = 1e-4
        self.learning_rate_decay_factor = 0.99 # don't use this decay
        self.max_gradient_norm = 5.0

        self.data_dir = 'data/'
        self.ckpt_dir = 'checkpoints/'
        self.save_every =  max(1, self.num_epochs//4) # save every 500 epochs

def create_model(sess, FLAGS):

    if FLAGS.model_name == 'GRU-LN':
        fw_model = gru_model(FLAGS)
    else:
        fw_model = fast_weights_model(FLAGS)

    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Restoring old model parameters from %s" %
                             ckpt.model_checkpoint_path)
        fw_model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created new model.")
        sess.run(tf.initialize_all_variables())

    return fw_model

def train(FLAGS):
    """
    Train the model on the associative retrieval task.
    """

    # Load the train/valid datasets
    print "Loading datasets:"
    with open(os.path.join(FLAGS.data_dir, 'train.p'), 'rb') as f:
        train_X, train_y = cPickle.load(f)
        print "train_X:", np.shape(train_X), ",train_y:", np.shape(train_y)
    with open(os.path.join(FLAGS.data_dir, 'valid.p'), 'rb') as f:
        valid_X, valid_y = cPickle.load(f)
        print "valid_X:", np.shape(valid_X), ",valid_y:", np.shape(valid_y)

    with tf.Session() as sess:

        # Load the model
        model = create_model(sess, FLAGS)
        start_time = time.time()

        # Start training
        train_epoch_loss = []; valid_epoch_loss = []
        train_epoch_accuracy = []; valid_epoch_accuracy = []
        train_epoch_gradient_norm = []
        for train_epoch_num, train_epoch in enumerate(generate_epoch(
            train_X, train_y, FLAGS.num_epochs, FLAGS.batch_size)):
            print "EPOCH:", train_epoch_num

            # Assign the learning rate
            sess.run(tf.assign(model.lr, FLAGS.learning_rate))

            #sess.run(tf.assign(model.lr, FLAGS.learning_rate))
            # Decay the learning rate
            #sess.run(tf.assign(model.lr, FLAGS.learning_rate * \
            #    (FLAGS.learning_rate_decay_factor ** epoch_num)))

            #if epoch_num < 1000:
            #    sess.run(tf.assign(model.lr, FLAGS.learning_rate))
            #elif epoch_num >= 1000: # slow down now
            #    sess.run(tf.assign(model.lr, 1e-4))

            # Custom decay (empirically decided)
            #if (epoch_num%1000 == 0):
            #    sess.run(tf.assign(model.lr,
            #        FLAGS.learning_rate/(10**(epoch_num//1000))))

            # Train set
            train_batch_loss = []
            train_batch_accuracy = []
            train_batch_gradient_norm = []
            for train_batch_num, (batch_X, batch_y) in enumerate(train_epoch):

                loss, accuracy, norm, _ = model.step(sess, batch_X, batch_y,
                    FLAGS.l, FLAGS.e, forward_only=False)
                train_batch_loss.append(loss)
                train_batch_accuracy.append(accuracy)
                train_batch_gradient_norm.append(norm)

            train_epoch_loss.append(np.mean(train_batch_loss))
            train_epoch_accuracy.append(np.mean(train_batch_accuracy))
            train_epoch_gradient_norm.append(np.mean(train_batch_gradient_norm))
            print ('Epoch: [%i/%i] time: %.4f, loss: %.7f,'
                    ' acc: %.7f, norm: %.7f' % (train_epoch_num, FLAGS.num_epochs,
                        time.time() - start_time, train_epoch_loss[-1],
                        train_epoch_accuracy[-1], train_epoch_gradient_norm[-1]))

            # Validation set
            valid_batch_loss = []
            valid_batch_accuracy = []
            for valid_epoch_num, valid_epoch in enumerate(generate_epoch(
                valid_X, valid_y, num_epochs=1, batch_size=FLAGS.batch_size)):

                for valid_batch_num, (batch_X, batch_y) in enumerate(valid_epoch):
                    loss, accuracy = model.step(sess, batch_X, batch_y,
                        FLAGS.l, FLAGS.e, forward_only=True)
                    valid_batch_loss.append(loss)
                    valid_batch_accuracy.append(accuracy)

            valid_epoch_loss.append(np.mean(valid_batch_loss))
            valid_epoch_accuracy.append(np.mean(valid_batch_accuracy))

            # Save the model
            if (train_epoch_num % FLAGS.save_every == 0 or
                train_epoch_num == (FLAGS.num_epochs-1)) and \
                (train_epoch_num > 0):
                if not os.path.isdir(FLAGS.ckpt_dir):
                    os.makedirs(FLAGS.ckpt_dir)
                checkpoint_path = os.path.join(FLAGS.ckpt_dir,
                    "%s.ckpt" % model_name)
                print "Saving the model."
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step)

        plt.plot(train_epoch_accuracy, label='train accuracy')
        plt.plot(valid_epoch_accuracy, label='valid accuracy')
        plt.legend(loc=4)
        plt.title('%s_Accuracy' % FLAGS.model_name)
        plt.show()

        plt.plot(train_epoch_loss, label='train loss')
        plt.plot(valid_epoch_loss, label='valid loss')
        plt.legend(loc=3)
        plt.title('%s_Loss' % FLAGS.model_name)
        plt.show()

        plt.plot(train_epoch_gradient_norm, label='gradient norm')
        plt.legend(loc=4)
        plt.title('%s_Gradient Norm' % FLAGS.model_name)
        plt.show()

        # Store results for global plot
        with open('%s_results.p' % FLAGS.model_name, 'wb') as f:
            cPickle.dump([train_epoch_accuracy, valid_epoch_accuracy,
                train_epoch_loss, valid_epoch_loss,
                train_epoch_gradient_norm], f)

def test(FLAGS):
    """
    Sample inputs of your own.
    """
    # Corpus for indexing
    corpus = ['a','b','c','d','e','f','g','h','i','j','k','l',
              'm','n','o','p','q','r','s','t','u','v','w','x','y','z',
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '?']

    # Render the sample to proper input format
    sample = 'g5o8k1??g'
    X = []
    for item in sample:
        X.append(corpus.index(item))
    X_one_hot = np.eye(26+10+1)[np.array(X).astype('int')]

    with tf.Session() as sess:

        if FLAGS.model_name == 'RNN-LN-FW':

            # Inputs need to real inputs of batch_size 128
            # because we use A(t) which updates even during testing

            # Load the model
            model = create_model(sess, FLAGS)

            # Load real samples
            with open(os.path.join(FLAGS.data_dir, 'train.p'), 'rb') as f:
                train_X, train_y = cPickle.load(f)
            for train_epoch_num, train_epoch in enumerate(generate_epoch(
                train_X, train_y, 1, FLAGS.batch_size)):
                for train_batch_num, (batch_X, batch_y) in enumerate(train_epoch):
                    batch_X[0] = X_one_hot
                    logits = model.logits.eval(feed_dict={model.X: batch_X,
                        model.l: FLAGS.l, model.e: FLAGS.e})

                    print "INPUT:", sample
                    print "PREDICTION:", corpus[np.argmax(logits[0])]

                    return

        else:
            # Reset from train sizes to sample sizes
            FLAGS.batch_size = 1

            # Load the model
            model = create_model(sess, FLAGS)
            logits = model.logits.eval(feed_dict={model.X: [X_one_hot],
                model.l: FLAGS.l, model.e: FLAGS.e})

            print "INPUT:", sample
            print "PREDICTION:", corpus[np.argmax(logits)]


def plot_all():
    """
    Plot the results.
    """

    with open('CONTROL_results.p', 'rb') as f:
        control_results = cPickle.load(f)
    with open('RNN-LN_results.p', 'rb') as f:
        RNN_LN_results = cPickle.load(f)
    with open('RNN-LN-FW_results.p', 'rb') as f:
        RNN_LN_FW_results = cPickle.load(f)
    with open('GRU-LN_results.p', 'rb') as f:
        GRU_LN_results = cPickle.load(f)

    # Plotting accuracy
    fig = plt.figure()
    plt.plot(control_results[1], label='Control accuracy')
    plt.plot(RNN_LN_results[1], label='RNN-LN accuracy')
    plt.plot(RNN_LN_FW_results[1], label='RNN-LN-FW accuracy')
    plt.plot(GRU_LN_results[1], label='GRU-LN accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc=4)
    fig.savefig('accuracy.png')
    #plt.show()

    # Plotting loss
    fig = plt.figure()
    plt.plot(control_results[3], label='Control loss')
    plt.plot(RNN_LN_results[3], label='RNN-LN loss')
    plt.plot(RNN_LN_FW_results[3], label='RNN-LN-FW loss')
    plt.plot(GRU_LN_results[3], label='GRU-LN loss')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=1)
    fig.savefig('loss.png')
    #plt.show()


if __name__ == '__main__':

    FLAGS = parameters()

    if sys.argv[1] == 'train':
        model_name = sys.argv[2]
        FLAGS.ckpt_dir = FLAGS.ckpt_dir + model_name
        FLAGS.model_name = model_name
        train(FLAGS)
    elif sys.argv[1] == 'test':
        model_name = sys.argv[2]
        FLAGS.ckpt_dir = FLAGS.ckpt_dir + model_name
        FLAGS.model_name = model_name
        test(FLAGS)
    elif sys.argv[1] == 'plot':
        plot_all()



