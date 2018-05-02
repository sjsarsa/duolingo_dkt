import numpy as np
from collections import defaultdict
from eval import eval

def get_word_counts(filename):
    word_counts = defaultdict(int)
    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()
            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue\n",
            if len(line) == 0:
                continue
            # If the line starts with #, then we're beginning a new exercise\n",
            elif line[0] == '#':
                continue
            else:
                line = line.split()
                word_counts[line[1].lower()] += 1
    return word_counts

def load_data(filename, word_counts):
    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.

    Returns:
        data: a list of InstanceData objects from that data type and track.
        labels (optional): if you specified training data, a dict of instance_id:label pairs.
    """

    # 'data' stores a list of 'InstanceData's as values.
    data = []

    # If this is training data, then 'labels' is a dict that contains instance_ids as keys and labels as values.
    training = False
    if filename.find('train') != -1:
        training = True

    if training:
        labels = dict()

    num_exercises = 0
    print('Loading instances...')

    def word_len(word):
        n = len(word)
        if n < 4: return 'short'
        if n < 7: return 'medium'
        if n < 10: return 'med_long'
        else: return 'long'
    # Compute word counts

    max_word_count = np.max(list(word_counts.values()))
    freqs = defaultdict(int)

    def word_freq(token):
        frequency = word_counts[token.lower()] / max_word_count
        if frequency < .001: freqs['<.001'] += 1; return 'very_rare'
        if frequency < .005: freqs['<.005'] += 1; return 'rare'
        if frequency < .01: freqs['<.01'] += 1; return 'medium_rare'
        if frequency < .05: freqs['<.05'] += 1; return 'semi_rare'
        if frequency < .1: freqs['<.1'] += 1;return 'quite_common'
        if frequency < .5: freqs['<.5'] += 1; return 'common'
        else: freqs['>=.5'] += 1; return 'very_common'

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()

            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:
                num_exercises += 1
                if num_exercises % 100000 == 0:
                    print('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...')

            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                list_of_exercise_parameters = line[2:].split()
                instance_properties = dict()
                for exercise_parameter in list_of_exercise_parameters:
                    [key, value] = exercise_parameter.split(':')
                    if key == 'countries':
                        value = value.split('|')
                    elif key == 'days':
                        value = float(value)
                    elif key == 'time':
                        if value == 'null':
                            value = 0#None
                        else:
                            assert '.' not in value
                            value = int(value)
                    instance_properties[key] = value

            # Otherwise we're parsing a new Instance for the current exercise
            else:
                line = line.split()
                if training:
                    assert len(line) == 7
                else:
                    assert len(line) == 6
                assert len(line[0]) == 12

                instance_properties['instance_id'] = line[0]

                instance_properties['token'] = line[1]
                instance_properties['frequency'] = word_freq(line[1])
                instance_properties['length'] = word_len(line[1])
                instance_properties['part_of_speech'] = line[2]

                instance_properties['morphological_features'] = dict()
                for l in line[3].split('|'):
                    [key, value] = l.split('=')
                    if key == 'Person':
                        value = int(value)
                    instance_properties['morphological_features'][key] = value

                instance_properties['dependency_label'] = line[4]
                instance_properties['dependency_edge_head'] = int(line[5])
                if training:
                    label = float(line[6])
                    labels[instance_properties['instance_id']] = label

                instance_properties['word_counts'] = word_counts
                data.append(InstanceData(instance_properties=instance_properties))

        print('Done loading ' + str(len(data)) + ' instances across ' + str(num_exercises) +
              ' exercises.\n')
    print(freqs)
    if training:
        return data, labels
    else:
        return data


class InstanceData(object):
    """
    A bare-bones class to store the included properties of each instance. This is meant to act as easy access to the
    data, and provides a launching point for deriving your own features from the data.
    """

    def __init__(self, instance_properties):
        # Parameters specific to this instance
        self.instance_id = instance_properties['instance_id']
        self.token = instance_properties['token']
        self.length = instance_properties['length']
        self.frequency = instance_properties['frequency']
        self.part_of_speech = instance_properties['part_of_speech']
        self.morphological_features = instance_properties['morphological_features']
        self.dependency_label = instance_properties['dependency_label']
        self.dependency_edge_head = instance_properties['dependency_edge_head']

        # Derived parameters specific to this instance
        self.exercise_index = int(self.instance_id[8:10])
        self.token_index = int(self.instance_id[10:12])

        # Derived parameters specific to this exercise
        self.exercise_id = self.instance_id[:10]

        # Parameters shared across the whole session
        self.user = instance_properties['user']
        self.countries = instance_properties['countries']
        self.days = instance_properties['days']
        self.client = instance_properties['client']
        self.session = instance_properties['session']
        self.format = instance_properties['format']
        self.time = instance_properties['time']

        # Derived parameters shared across the whole session
        self.session_id = self.instance_id[:8]

        self.word_counts = instance_properties['word_counts']
    def to_features(self):
        """
        Prepares those features that we wish to use in the LogisticRegression example in this file. We introduce a bias,
        and take a few included features to use. Note that this dict restructures the corresponding features of the
        input dictionary, 'instance_properties'.

        Returns:
            to_return: a representation of the features we'll use for logistic regression in a dict. A key/feature is a
                key/value pair of the original 'instance_properties' dict, and we encode this feature as 1.0 for 'hot'.
        """
        to_return = dict()

        to_return['bias'] = 1.0
        to_return['user:' + self.user] = 1.0
        to_return['format:' + self.format] = 1.0
        to_return['token:' + self.token.lower()] = 1.0
        to_return['frequency:' + self.frequency] = 1.0
        to_return['frequency'] = self.word_counts[self.token.lower()] / 55197 # normalize by max word count
        to_return['length:' + self.length] = 1.0
        to_return['length'] = len(self.token) / 15 # normalize by max word length
        to_return['time'] = np.abs(self.time) / 30
        to_return['session' + self.session] = 1.0
        to_return['part_of_speech:' + self.part_of_speech] = 1.0
        for morphological_feature in self.morphological_features:
            to_return['morphological_feature:' + morphological_feature] = 1.0
        to_return['dependency_label:' + self.dependency_label] = 1.0

        return to_return


def init_feature_map(data, feature_dict, max_users=float('inf')):
    """
    maps each distinct feature to a unique index for vectorization
    restricted by given amount of users

    :param data: instance data as a list
    :param feature_dict: dict in which to map the indices
    :param max_users: defaults to infinity, restrict amount of users for limited RAM and faster learning
    :return: nothing, this isn't functional
    """
    i = 0
    users = 0
    for instance_data in data:
        new_user = False
        for key in instance_data.to_features().keys():
            if key not in feature_dict and 'user:' in key:
                new_user = True
                users += 1
        if new_user and users > max_users: break
        for key in instance_data.to_features().keys():
            if key not in feature_dict:
                feature_dict[key] = i
                # print('Mapped feature', key, 'to', i)
                i += 1
    print('Total features:', len(feature_dict))


def vectorize_features(data, onehot_feature_map, training_labels=None):
    """
    Transforms one hot data in dicts to vectors according to given map,
    users are restricted according to the map.
    If there are labels for the data, it is filtered according to the users in the map.
    :param data:
    :param onehot_feature_map:
    :param training_labels:
    :return:
    """
    formatted_instances = []
    labels = []
    included_instances = 0
    n = len(onehot_feature_map)
    for instance_i, instance_data in enumerate(data):
        categorical_vec = np.zeros(n)
        excluded_user = False
        for key in instance_data.to_features().keys():
            if key not in onehot_feature_map:
                excluded_user = True
                break
            categorical_vec[onehot_feature_map[key]] = 1

        if (instance_i + 1) % 100000 == 0: print('\r', round((instance_i + 1) / len(data), 3) * 100,
                                                 '% of instances processed', end='')
        if not excluded_user:
            formatted_instances.append(categorical_vec)
            if training_labels: labels.append(training_labels[instance_data.instance_id])
            included_instances += 1
            # print(categorical_vec)
    print('100 % of instances processed')
    print(included_instances, 'instances included,', len(data) - included_instances, 'excluded')
    if training_labels: return np.matrix(formatted_instances), np.array(labels)
    return np.matrix(formatted_instances)


def evaluate_predictions(pred_filename, predictions):
    with open(pred_filename, 'wt') as f:
        for instance_id, prediction in predictions.items():
            f.write(instance_id + ' ' + str(prediction) + '\n')

    eval(pred_filename, '../data_fr_en/fr_en.slam.20171218.dev.key')
