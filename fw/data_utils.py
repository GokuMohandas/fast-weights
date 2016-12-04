import numpy as np
import random
import cPickle
import os

"""
    Creating the data set for fast weights implementation.
    Data will minmic the synthetic dataset created in
    https://arxiv.org/abs/1610.06258 Ba et al.

    Ex.
    c6a7s4??a = 7 (it asking for the value for the key a)
    This is a very interesting dataset because
    it requires the model to retrieve and use temporary memory
    in order to accurately predict the proper value for the key.
"""

def get_three_letters():
    """
    Retrieve three random letters (a-z)
    without replacement.
    """
    return np.random.choice(range(0,26), 3, replace=False)

def get_three_numbers():
    """
    Retrieve three random numbers (0-9)
    with replacement.
    """
    return np.random.choice(range(26, 26+10), 3, replace=True)

def create_sequence():
    """
    Concatenate keys and values with
    ?? and one of the keys.
    Returns the input and output.
    """
    letters = get_three_letters()
    numbers = get_three_numbers()
    X = np.zeros((9))
    y = np.zeros((1))
    for i in range(0, 5, 2):
        X[i] = letters[i/2]
        X[i+1] = numbers[i/2]

    # append ??
    X[6] = 26+10
    X[7] = 26+10

    # last key and respective value (y)
    index = np.random.choice(range(0,3), 1, replace=False)
    X[8] = letters[index]
    y = numbers[index]

    # one hot encode X and y
    X_one_hot = np.eye(26+10+1)[np.array(X).astype('int')]
    y_one_hot = np.eye(26+10+1)[y][0]

    return X_one_hot, y_one_hot

def ordinal_to_alpha(sequence):
    """
    Convert from ordinal to alpha-numeric representations.
    Just for funsies :)
    """
    corpus = ['a','b','c','d','e','f','g','h','i','j','k','l',
              'm','n','o','p','q','r','s','t','u','v','w','x','y','z',
               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '?']

    conversion = ""
    for item in sequence:
        conversion += str(corpus[int(item)])
    return conversion

def create_data(num_samples):
    """
    Create a num_samples long set of X and y.
    """
    X = np.zeros([num_samples, 9, 26+10+1], dtype=np.int32)
    y = np.zeros([num_samples, 26+10+1], dtype=np.int32)
    for i in range(num_samples):
        X[i], y[i] = create_sequence()
    return X, y

def generate_epoch(X, y, num_epochs, batch_size):

    for epoch_num in range(num_epochs):
        yield generate_batch(X, y, batch_size)

def generate_batch(X, y, batch_size):

    data_size = len(X)

    num_batches = (data_size // batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield X[start_index:end_index], y[start_index:end_index]

if __name__ == '__main__':

    # Sampling
    sample_X, sample_y = create_sequence()
    print "Sample:", ordinal_to_alpha([np.argmax(X) for X in sample_X]), \
        ordinal_to_alpha([np.argmax(sample_y)])

    # Train/valid sets
    train_X, train_y = create_data(64000)
    print "train_X:", np.shape(train_X), ",train_y:", np.shape(train_y)
    valid_X, valid_y = create_data(32000)
    print "valid_X:", np.shape(valid_X), ",valid_y:", np.shape(valid_y)

    # Save data into pickle files
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/train.p', 'wb') as f:
        cPickle.dump([train_X, train_y], f)
    with open('data/valid.p', 'wb') as f:
        cPickle.dump([valid_X, valid_y], f)


