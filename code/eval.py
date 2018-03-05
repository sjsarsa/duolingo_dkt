from __future__ import division

import argparse
from io import open
import math
import os


def eval(key_file, pred_file):
    """
    Evaluates your predictions. This loads the dev labels and your predictions, and then evaluates them, printing the
    results for a variety of metrics to the screen.
    """

    test_metrics()

    print('\nLoading labels for exercises...')
    labels = load_labels(key_file)

    print('Loading predictions for exercises...')
    predictions = load_labels(pred_file)

    actual = []
    predicted = []

    missing_instances = 0
    for instance_id in labels.keys():
        try:
            predicted.append(predictions[instance_id])
            actual.append(labels[instance_id])
        except KeyError:
            missing_instances += 1
    print('number of instances:', len(actual))
    print('missing instances:', missing_instances)
    metrics = evaluate_metrics(actual, predicted)
    line = '\t'.join([('%s=%.3f' % (metric, value)) for (metric, value) in metrics.items()])
    print('Metrics:\t' + line)


def load_labels(filename):
    """
    This loads labels, either the actual ones or your predictions.

    Parameters:
        filename: the filename pointing to your labels

    Returns:
        labels: a dict of instance_ids as keys and labels between 0 and 1 as values
    """
    labels = dict()

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                line = line.split()
            instance_id = line[0]
            label = float(line[1])
            labels[instance_id] = label
    return labels


def compute_acc(actual, predicted):
    """
    Computes the accuracy of your predictions, using 0.5 as a cutoff.

    Note that these inputs are lists, not dicts; they assume that actual and predicted are in the same order.

    Parameters (here and below):
        actual: a list of the actual labels
        predicted: a list of your predicted labels
    """
    num = len(actual)
    acc = 0.
    for i in range(num):
        if round(actual[i], 0) == round(predicted[i], 0):
            acc += 1.
    acc /= num
    return acc


def compute_avg_log_loss(actual, predicted):
    """
    Computes the average log loss of your predictions.
    """
    num = len(actual)
    loss = 0.

    for i in range(num):
        p = predicted[i] if actual[i] > .5 else 1. - predicted[i]
        if p == 0: p += .00001
        loss -= math.log(p)
    loss /= num
    return loss


def compute_auroc(actual, predicted):
    """
    Computes the area under the receiver-operator characteristic curve.
    This code a rewriting of code by Ben Hamner, available here:
    https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
    """
    num = len(actual)
    temp = sorted([[predicted[i], actual[i]] for i in range(num)], reverse=True)

    sorted_predicted = [row[0] for row in temp]
    sorted_actual = [row[1] for row in temp]

    sorted_posterior = sorted(zip(sorted_predicted, range(len(sorted_predicted))))
    r = [0 for k in sorted_predicted]
    cur_val = sorted_posterior[0][0]
    last_rank = 0
    for i in range(len(sorted_posterior)):
        if cur_val != sorted_posterior[i][0]:
            cur_val = sorted_posterior[i][0]
            for j in range(last_rank, i):
                r[sorted_posterior[j][1]] = float(last_rank + 1 + i) / 2.0
            last_rank = i
        if i == len(sorted_posterior) - 1:
            for j in range(last_rank, i + 1):
                r[sorted_posterior[j][1]] = float(last_rank + i + 2) / 2.0

    num_positive = len([0 for x in sorted_actual if x == 1])
    num_negative = num - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if sorted_actual[i] == 1])
    auroc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) / (num_negative * num_positive))

    return auroc


def compute_f1(actual, predicted):
    """
    Computes the F1 score of your predictions. Note that we use 0.5 as the cutoff here.
    """
    num = len(actual)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for i in range(num):
        if actual[i] >= 0.5 and predicted[i] >= 0.5:
            true_positives += 1
        elif actual[i] < 0.5 and predicted[i] >= 0.5:
            false_positives += 1
        elif actual[i] >= 0.5 and predicted[i] < 0.5:
            false_negatives += 1
        else:
            true_negatives += 1

    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        F1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        F1 = 0.0

    return F1


def evaluate_metrics(actual, predicted):
    """
    This computes and returns a dictionary of notable evaluation metrics for your predicted labels.
    """
    acc = compute_acc(actual, predicted)
    avg_log_loss = compute_avg_log_loss(actual, predicted)
    auroc = compute_auroc(actual, predicted)
    F1 = compute_f1(actual, predicted)

    return {'accuracy': acc, 'avglogloss': avg_log_loss, 'auroc': auroc, 'F1': F1}


def test_metrics():
    actual = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
    predicted = [0.8, 0.2, 0.6, 0.3, 0.1, 0.2, 0.3, 0.9, 0.2, 0.7]
    metrics = evaluate_metrics(actual, predicted)
    metrics = {key: round(metrics[key], 3) for key in metrics.keys()}
    assert metrics['accuracy'] == 0.700
    assert metrics['avglogloss'] == 0.613
    assert metrics['auroc'] == 0.740
    assert metrics['F1'] == 0.667
    print('Verified that our environment is calculating metrics correctly.')


if __name__ == '__main__':
    main()
