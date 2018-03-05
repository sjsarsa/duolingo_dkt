from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Conv1D, MaxPooling1D
from keras import optimizers
from metrics import f1
import numpy as np


def predict_test_set(instance_names, predictions):
    print(len(instance_names))
    print(len(predictions))
    return {instance_name: prediction[0] for instance_name, prediction in zip(instance_names, predictions)}


"""
This isn't good for this task as a regular nn is not capable of inferring sequential properties, fun to test though
"""


def work_the_magic(X_train, Y_train, X_test, Y_test, names_test):
    model = Sequential([
        Dense(128, activation='relu', input_dim=X_train.shape[1]),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', f1])
    model.fit(X_train, Y_train, epochs=2)

    # scores = model.evaluate(X_test, Y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1]*100))
    print('Predicting instances...')
    predictions = model.predict(X_test)
    # print(predictions)
    # with open('nn_predictions_14_02.pred', 'w') as f:
    #     for prediction in predictions:
    #         f.write(str(prediction) + '\n')

    return predict_test_set(names_test, predictions)
