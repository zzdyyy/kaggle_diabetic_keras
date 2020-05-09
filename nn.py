"""create CNN model and train it"""

import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score, classification_report

import numpy as np
import scipy.io as sio
from pprint import pprint

# Tensorflow here is used only in keras metric function 'kappa'. If you use other
# keras backend, you can rewrite the kappa in this module.
import tensorflow as tf

import data
import util
import iterator


def create_net(config, initial_epoch=0):
    model = keras.Sequential(name=config.get('name'))
    for layer, kwargs in config.layers:
        model.add(layer(**kwargs))
    model.compile(
        optimizer=SGD(nesterov=True, momentum=0.9,
                      lr=get_init_lr(config.get('schedule'), initial_epoch)
                      ),
        loss='mse',  # mean squared error
        metrics=['mse', 'mae', kappa],
    )
    return model


def kappa(y_true, y_pred):
    return tf.py_func(util.kappa, [y_true, y_pred], tf.float32, stateful=False)


def train_model(model: keras.Sequential, config, initial_epoch, files, labels):
    # prepare train/validation data generator
    eval_size = 0.1
    X_train, X_valid, y_train, y_valid = train_test_split(files, labels, eval_size)  # still not image
    batch_size_train = config.get('batch_size_train')
    batch_size_valid = config.get('batch_size_test')
    train_steps = (X_train.shape[0] + batch_size_train - 1) // batch_size_train
    valid_steps = (X_valid.shape[0] + batch_size_valid - 1) // batch_size_valid
    # when feed with files and labels, these class returns images/labels data once
    batch_iterator_train = iterator.ResampleIterator(config, batch_size=batch_size_train, initial_epoch=initial_epoch)
    batch_iterator_valid = iterator.SharedIterator(config, batch_size=batch_size_valid, deterministic=True)
    # add a adapter generator who can yield data infinitely
    train_iter = generator_adapter(X_train, y_train, batch_iterator_train)
    valid_iter = generator_adapter(X_valid, y_valid, batch_iterator_valid)

    # set up callbacks for saving weight and scheduling learning-rate
    callbacks = [
        ModelCheckpoint(verbose=1,
                        filepath=config.weights_epoch,
                        period=5
                        ),  # save every 5 epochs
        ModelCheckpoint(verbose=1,
                        filepath=config.weights_best,
                        period=1,
                        save_best_only=True
                        ),  # save best loss
        ModelCheckpoint(verbose=1,
                        filepath=config.weights_file,
                        monitor='val_kappa',
                        period=1,
                        save_best_only=True,
                        mode='max',
                        ),  # save best kappa
        LearningRateScheduler(verbose=1,
                              schedule=lr_scheduler(config.get('schedule'))
                              )
    ]
    epochs = get_max_epoch(config.get('schedule'))

    model.fit_generator(
        generator=train_iter,
        steps_per_epoch=train_steps,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=valid_iter,
        validation_steps=valid_steps,
    )


def train_test_split(X, y, eval_size):
    if eval_size:
        X_train, X_valid, y_train, y_valid = data.split(
            X, y, test_size=eval_size)
    else:
        X_train, y_train = X, y
        X_valid, y_valid = X[len(X):], y[len(y):]

    return X_train, X_valid, y_train, y_valid


def generator_adapter(X, y, iterobj):
    while True:
        for Xs, ys in iterobj(X, y):
            Xs = Xs.transpose([0,3,2,1])
            yield(Xs, ys)


def generator_adapter_test(X, y, iterobj):
    while True:
        for Xs, ys in iterobj(X, y):
            Xs = Xs.transpose([0,3,2,1])
            yield Xs


def get_max_epoch(scheme):
    # e.g. scheme = {0:0.003, 150:0.0003, 201:'stop'}
    for key, val in scheme.items():
        if val == 'stop':
            return max(0, key)
    else:
        raise(ValueError('no "stop" found in schedule configuration.'))


def lr_scheduler(scheme):
    # e.g. scheme = {0:0.003, 150:0.0003, 201:'stop'}

    def calculate_lr(epoch, lr):
        if epoch in scheme:
            return scheme[epoch]
        else:
            return lr

    return calculate_lr


def get_init_lr(scheme, epoch):
    # e.g. scheme = {0:0.003, 150:0.0003, 201:'stop'}
    lr = 0.001
    for e, l in scheme.items():
        if epoch >= e:
            lr = l
    return lr


def test_model(model: keras.Sequential, config, files, labels):
    X_test, y_test = files, labels  # still not image
    batch_size_test = config.get('batch_size_test')
    test_steps = (X_test.shape[0] + batch_size_test - 1) // batch_size_test
    batch_iterator_test = iterator.SharedIterator(config, batch_size=batch_size_test, deterministic=True)
    test_iter = generator_adapter_test(X_test, y_test, batch_iterator_test)

    y_pred = model.predict_generator(test_iter, test_steps, verbose=1)
    # pprint(list(zip(files, y_pred)))
    for filename, score in zip(files, y_pred):
        print(filename, score, sep='\t')


def eval_model(model: keras.Sequential, config, files, labels):
    X_test, y_test = files, labels  # still not image
    batch_size_test = config.get('batch_size_test')
    test_steps = (X_test.shape[0] + batch_size_test - 1) // batch_size_test
    batch_iterator_test = iterator.SharedIterator(config, batch_size=batch_size_test, deterministic=True)
    test_iter = generator_adapter_test(X_test, y_test, batch_iterator_test)

    y_test = keras.utils.to_categorical(y_test, 5)
    y_pred = model.predict_generator(test_iter, test_steps, verbose=1)
    # pprint(list(zip(files, y_pred)))
    print('kappa: ', util.kappa(y_test, y_pred))
    y_gt_cat = np.argmax(y_test, -1)
    y_pred_cat = np.round(np.clip(y_pred, 0., 4.)).astype(int)# np.argmax(y_pred, -1)
    print('accuracy:', accuracy_score(y_gt_cat, y_pred_cat))
    print(classification_report(y_gt_cat, y_pred_cat))
