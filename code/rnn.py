from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed, LSTM, Conv1D, MaxPooling1D, Dropout, Masking
from keras import optimizers
import numpy as np
from sklearn.metrics import roc_auc_score
import metrics


def copy_in_steps(ar1, ar2, steps):
    """
    Use in order to prevent memory error
    """
    step_size = int(len(ar1) / steps)
    for i in range(steps):
        ar1[i*step_size:(i+1)*step_size] = np.copy(ar2[i*step_size:(i+1)*step_size])
    if steps * step_size < len(ar1): ar1[steps * step_size:] = np.copy(ar2[steps * step_size:])


def split_to_sequences(X, chunk_size, batch_size, i=0, steps=10):
    """
    Pad with zeros to enable even split, then split
    """
    while (X.shape[0] + i) % chunk_size != 0 or ((X.shape[0] + i) / chunk_size) % batch_size != 0:
        i += 1
    padded = np.zeros((X.shape[0] + i, X.shape[1]))
    print(padded.shape)
    copy_in_steps(padded[:X.shape[0]], X, steps)
    return np.reshape(padded, (-1, chunk_size, X.shape[1]))


def predict_test_set(instance_names, predictions):
    return {instance_name: prediction for instance_name, prediction in zip(instance_names, predictions)}


def work_the_magic(X_train, Y_train, X_test, Y_test, names_test):
    """
    Simple keras lstm
    """
    # Todo: split users into sequences of size max(len(exercises_by_user_x)), this could be done elsewhere
    batch_size = 10
    time_window = 200
    X = split_to_sequences(X_train, time_window, batch_size)
    Y = split_to_sequences(np.expand_dims(Y_train, 2), time_window, batch_size, len(X) * time_window - len(X_train))

    model = Sequential([
        Masking(mask_value=0, input_shape=X.shape[1:], batch_input_shape=(batch_size, *X.shape[1:])),
        # Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(X.shape[1:])),
        LSTM(256, return_sequences=True, stateful=True),
        TimeDistributed(Dense(1, activation='sigmoid')),
        Dropout(.4)
    ])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy', metrics.f1, metrics.roc_auc])
    model.fit(X, Y, epochs=110, batch_size=batch_size)

    X_t = split_to_sequences(X_test, time_window, batch_size)

    predictions = model.predict(X_t, batch_size=batch_size)
    print(predictions.shape)

    return predict_test_set(names_test,
                            np.reshape(predictions, predictions.shape[0] * predictions.shape[1]))
