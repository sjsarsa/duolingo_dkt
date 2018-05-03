from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, TimeDistributed, LSTM, Bidirectional, Dropout, Masking
from keras import optimizers
from time import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from keras.callbacks import Callback
import metrics
from data import evaluate_predictions
from sklearn.preprocessing import label_binarize as binarize

def copy_in_steps(ar1, ar2, steps):
    """
    Use in order to prevent memory error
    """
    step_size = int(len(ar1) / steps)
    for i in range(steps):
        ar1[i * step_size:(i + 1) * step_size] = np.copy(ar2[i * step_size:(i + 1) * step_size])
    if steps * step_size < len(ar1): ar1[steps * step_size:] = np.copy(ar2[steps * step_size:])


def split_to_sequences(X, time_window, batch_size, i=0, steps=10):
    """
    Pad with negative ones to enable even split, then split
    """
    while (X.shape[0] + i) % time_window != 0 or ((X.shape[0] + i) / time_window) % batch_size != 0:
        i += 1
    padded = np.ones((X.shape[0] + i, X.shape[1])) * -1
    print(padded.shape)
    copy_in_steps(padded[:X.shape[0]], X, steps)
    return np.reshape(padded, (-1, time_window, X.shape[1]))


def predict_test_set(instance_names, predictions):
    return {instance_name: prediction for instance_name, prediction in zip(instance_names, predictions)}


def work_the_magic(X_train, Y_train, X_test, Y_test, names_test, sequence_size=8000, layer_size=400):
    """
    Simple keras lstm
    """
    batch_size = 10
    time_window = 100
    X = split_to_sequences(X_train, time_window, batch_size)
    Y = split_to_sequences(np.expand_dims(Y_train, 2), time_window, batch_size, len(X) * time_window - len(X_train))
    # X = np.reshape(X_train, (-1, time_window, X_train.shape[1]))# for state resetting model
    # Y = np.reshape(np.expand_dims(Y_train, 2), (-1, time_window, 1))# for state resetting model
    
    print('layer size', layer_size)
    model = Sequential([
        Masking(mask_value=-1, batch_input_shape=(batch_size, time_window, X_train.shape[-1])),
        Bidirectional(LSTM(layer_size, return_sequences=True, stateful=True, dropout=.1)),
        TimeDistributed(Dense(1, activation='sigmoid')),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])

    # model.fit(X, Y, epochs=20, batch_size=batch_size)

    def evaluate_model():
        X_t = split_to_sequences(X_test, time_window, batch_size)
        preds = model.predict(X_t, batch_size=batch_size)
        predictions = np.reshape(preds, preds.shape[0] * preds.shape[1])
        print('Test predictions')
        evaluate_predictions('tmp_pred.pred',
                             predict_test_set(names_test, predictions))



    class EvalCallback(Callback):
        def __init__(self, interval=2):
            super(Callback, self).__init__()
            self.interval = interval

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.interval == 0:
                evaluate_model()

    eval_cb = EvalCallback()
    model.fit(X, Y, epochs=30, batch_size=batch_size, callbacks=[eval_cb], verbose=2)
        
    evaluate_model()

 #   for e in range(1, 100):
 #       print("Running epoch:", e)
 #       loss, acc = train_in_batches(X, Y, Y_train, model, batch_size)
 #       print('\rEpoch {} loss={:.5}, accuracy={:.5}'.format(e, loss, acc))
 #       #print('\rEpoch {} loss={:.5}, accuracy={:.5}, auc={:.5}, f1={:.5}'.format(e, loss, acc, auc, f1))
 #       if e % 5 == 0: evaluate_model()


def train_in_batches(X, Y, Y_t, model, batch_size):
    Y_flat = np.reshape(Y_t, -1)
    accs = []
    losses = []
    for i, b in enumerate(range(0, len(X), batch_size)):
        start = time()
        loss, acc = model.train_on_batch(X[b:b + batch_size], Y[b:b + batch_size])
        accs.append(acc)
        losses.append(loss)
              
        eta = (time() - start) * (len(X) - b) / batch_size
        print('\rloss={:.5} accuracy={:.5} remaining={:.3}% ETA={:.6}s'
              .format(loss, acc, (len(X) - b) / len(X), eta), end='')

    #preds = np.reshape(model.predict(X, batch_size=batch_size), -1)

    return np.mean(losses), np.mean(accs) #, roc_auc_score(Y_flat, preds[:len(Y_flat)]), f1_score(Y_flat, np.round(preds[:len(Y_flat)]))



# Training with resetting states between different user data, requires padding for each user sequence.
# NOT WORKING finish maybe later if there is enough time
def run_epoch(X, Y, t_X, t_Y, model, sequence_size, time_window, batch_size, e):
    e_start = time()
    # print('Train...')
    mean_tr_acc = []
    mean_tr_loss = []
    mean_tr_f1 = []
    mean_tr_auc = []
    # train in batches
    for b in range(0, len(X), sequence_size * batch_size):
        X_b = np.zeros((batch_size, sequence_size, X.shape[1]))
        Y_b = np.zeros((batch_size, sequence_size, 1))
        for i, s in enumerate(range(b, b + sequence_size * batch_size, sequence_size)):
            X_b[i, :, :] = X[s:s + sequence_size]
            Y_b[i, :, :] = np.expand_dims(Y[s:s + sequence_size], 2)
        # print('Xb shape:', X_b.shape, end=' ')
        if len(X_b.shape) <= 1: break
        # print('Yb shape:', Y_b.shape)
        if X_b[0, 0, 0] == -1: break
        for t in range(0, sequence_size, time_window):
            start = time()
            if X_b[0, t, 0] == -1: break
            tr_loss, tr_acc, tr_f1, tr_auc = model.train_on_batch(X_b[:, t:(t + time_window), :],
                                                                  Y_b[:, t:(t + time_window), :])
            mean_tr_acc.append(tr_acc)
            mean_tr_loss.append(tr_loss)
            mean_tr_f1.append(tr_f1)
            mean_tr_auc.append(tr_auc)
            eta = (time() - start) * ((len(X) - b - t - time_window) / time_window)
            print('\rloss={:.5} accuracy={:.5} f1={:.5} auc={:.5} remaining={:.3}% ETA={:.6}s\r'.format(
                tr_loss, tr_acc, tr_f1, tr_auc, (len(X) - b) / len(X), eta), end='')
        model.reset_states()
    print()
    print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
    print('loss training = {}'.format(np.mean(mean_tr_loss)))
    print('f1 training = {}'.format(np.nanmean(mean_tr_f1)))
    print('auc training = {}'.format(np.nanmean(mean_tr_auc)))
    print('___________________________________')

    if (e % 5 == 0):
        # print('Test...')
        mean_te_acc = []
        mean_te_loss = []
        mean_te_f1 = []
        mean_te_auc = []
        for b in range(0, len(X), sequence_size * batch_size):
            X_b = np.zeros((batch_size, sequence_size, t_X.shape[1]))
            Y_b = np.zeros((batch_size, sequence_size, 1))
            for i, s in enumerate(range(b, b + sequence_size * batch_size, sequence_size)):
                X_b[i, :, :] = t_X[s:s + sequence_size]
                Y_b[i, :, :] = np.expand_dims(t_Y[s:s + sequence_size], 2)
            if len(X_b.shape) <= 1: break
            if X_b[0, 0, 0] == -1: break
            for t in range(0, sequence_size, time_window):
                if X_b[0, t, 0] == -1: break
                te_loss, te_acc, te_f1, te_auc = model.test_on_batch(X_b[:, t:(t + time_window), :],
                                                                     Y_b[:, t:(t + time_window), :])
                mean_te_acc.append(te_acc)
                mean_te_loss.append(te_loss)
                mean_te_f1.append(te_f1)
                mean_te_auc.append(te_auc)
                print('\rloss=', te_loss, 'accuracy=', te_acc, 'auc=', te_auc,
                      'remaining', (len(X) - b) / len(X), '%', end='')
            model.reset_states()
        print()
        print('accuracy test = {}'.format(np.mean(mean_te_acc)))
        print('loss test = {}'.format(np.mean(mean_te_loss)))
        print('f1 test = {}'.format(np.nanmean(mean_te_f1)))
        print('auc test = {}'.format(np.nanmean(mean_te_auc)))
        print('Epoch time: {}s'.format(time() - e_start))
        print('___________________________________')

# def predict(X, model, sequence_size, time_window, batch_size):
#     predictions = []
#     print('Test...')
#     for b in range(0, len(X), sequence_size * batch_size):
#         X_b = np.zeros((batch_size, sequence_size, t_X.shape[1]))
#         Y_b = np.zeros((batch_size, sequence_size, 1))
#         for i, s in enumerate(range(b, b + sequence_size * batch_size, sequence_size)) :
#             X_b[i,:,:] = t_X[s:s+sequence_size]
#             Y_b[i,:,:] = np.expand_dims(t_Y[s:s+sequence_size], 2)
#         if len(X_b.shape) <= 1: break
#         if X_b[0,0,0] == -1: break
#         for t in range(0, sequence_size, time_window):
#             if X_b[0,t,0] == -1: break
#             predictions += model.test_on_batch(X_b[:,t:(t+time_window),:], Y_b[:,t:(t+time_window),:])
#         model.reset_states()
#     print()
#     print('accuracy test = {}'.format(np.mean(mean_te_acc)))
#     print('loss test = {}'.format(np.mean(mean_te_loss)))
#     print('f1 test = {}'.format(np.mean(mean_te_f1)))
#     print('auc test = {}'.format(np.mean(mean_te_auc)))
#     print('Epoch time: {}s'.format(time() - e_start))
#     print('___________________________________')
