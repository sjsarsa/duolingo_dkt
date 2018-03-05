"""
Duolingo SLAM Shared Task - Baseline Model

This baseline model loads the training and test data that you pass in via --train and --test arguments for a particular
track (course), storing the resulting data in InstanceData objects, one for each instance. The code then creates the
features we'll use for logistic regression, storing the resulting LogisticRegressionInstance objects, then uses those to
train a regularized logistic model with SGD, and then makes predictions for the test set and dumps them to a CSV file
specified with the --pred argument, in a format appropriate to be read in and graded by the eval.py script.

We elect to use two different classes, InstanceData and LogisticRegressionInstance, to delineate the boundary between
the two purposes of this code; the first being to act as a user-friendly interface to the data, and the second being to
train and run a baseline model as an example. Competitors may feel free to use InstanceData in their own code, but
should consider replacing the LogisticRegressionInstance with a class more appropriate for the model they construct.

This code is written to be compatible with both Python 2 or 3, at the expense of dependency on the future library. This
code does not depend on any other Python libraries besides future.
"""

import argparse
from io import open
import os
from data import InstanceData, load_data, vectorize_features, init_feature_map
import nn
import rnn
from eval import eval


def main():
    """
    Executes the baseline model. This loads the training data, training labels, and dev data, then trains a logistic
    regression model, then dumps predictions to the specified file.

    Modify the middle of this code, between the two commented blocks, to create your own model.
    """

    parser = argparse.ArgumentParser(description='Duolingo shared task baseline model')
    parser.add_argument('--train', help='Training file name', default='../data_fr_en/fr_en.slam.20171218.train')
    parser.add_argument('--test', help='Test file name, to make predictions on',
                        default='../data_fr_en/fr_en.slam.20171218.dev')
    parser.add_argument('--pred', help='Output file name for predictions, defaults to rnn_predictions_{n_users}.pred')
    args = parser.parse_args()

    assert os.path.isfile(args.train)
    assert os.path.isfile(args.test)

    # Assert that the train course matches the test course
    assert os.path.basename(args.train)[:5] == os.path.basename(args.test)[:5]

    training_data, training_labels = load_data(args.train)
    test_data = load_data(args.test)

    users = 1

    if not args.pred:
        args.pred = 'rnn_predictions_' + str(users) + '.pred'

    print('Using data for', users, 'users.')
    # create binary vectors of one hotted features
    print('Initializing one hot map for formatting the data...')
    feature_map = {}
    init_feature_map(training_data, feature_map, users)
    init_feature_map(test_data, feature_map, 0)
    print('One hot map initialized')
    print('Formatting train data...')
    X, Y = vectorize_features(training_data, feature_map, training_labels)
    print('Formatting test data...')
    X_t = vectorize_features(test_data, feature_map)
    # n = len(X)
    # n_train = int(0.8*n)
    # X_train = X[:n_train]
    # Y_train = Y[:n_train]
    # X_test = X[n_train:]
    # Y_test = Y[n_train:]
    print('Collecting test labels...')
    names_test = [instance_data.instance_id for instance_data in test_data]

    # predictions = nn.work_the_magic(X_train, Y_train,
    #                                  X_test, Y_test, names_test)
    predictions = rnn.work_the_magic(X, Y, X_t, [], names_test)

    print('Printing predictions...')
    with open(args.pred, 'wt') as f:
        for instance_id, prediction in predictions.items():
            f.write(instance_id + ' ' + str(prediction) + '\n')
    print('Printing finished')

    eval('../data_fr_en/fr_en.slam.20171218.dev.key', args.pred)


if __name__ == '__main__':
    main()
