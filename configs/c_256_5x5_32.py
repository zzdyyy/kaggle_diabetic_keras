import numpy as np

from config import Config
from data import BALANCE_WEIGHTS
from layers import *

cnf = {
    'name': __name__.split('.')[-1],
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_small',
    'test_dir': 'data/test_small',
    'batch_size_train': 128,
    'batch_size_test': 16,
    'balance_weights': np.array(BALANCE_WEIGHTS),
    'balance_ratio': 0.975,
    'final_balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.15, 1.15),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'sigma': 0.5,
    'schedule': {
        0: 0.003,
        150: 0.0003,
        201: 'stop',
    },
}

n = 32

layers = [
    (InputLayer, {'input_shape': (cnf['h'], cnf['w'], 3)}),
    (Conv2D, conv_params(n, kernel_size=(5, 5), strides=(2, 2),
                kernel_regularizer=l1_l2(regular_factor_l1, regular_factor_l2),
                bias_regularizer=l1_l2(regular_factor_l1, regular_factor_l2))),
    (Conv2D, conv_params(n)),
    (MaxPool2D, pool_params()),
    (Conv2D, conv_params(2 * n, kernel_size=(5, 5), strides=(2, 2))),
    (Conv2D, conv_params(2 * n)),
    (Conv2D, conv_params(2 * n)),
    (MaxPool2D, pool_params()),
    (Conv2D, conv_params(4 * n)),
    (Conv2D, conv_params(4 * n)),
    (Conv2D, conv_params(4 * n)),
    (MaxPool2D, pool_params()),
    (Conv2D, conv_params(8 * n)),
    (Conv2D, conv_params(8 * n)),
    (Conv2D, conv_params(8 * n)),
    (RMSPoolLayer, pool_params(strides=(3, 3))),
    (Dropout, {'rate': 0.5}),
    (Flatten, {}), (Dense, dense_params(1024)),
    (Reshape, {'target_shape': (-1, 1)}), (MaxPooling1D, {'pool_size': 2}),
    (Dropout, {'rate': 0.5}),
    (Flatten, {}), (Dense, dense_params(1024)),
    (Reshape, {'target_shape': (-1, 1)}), (MaxPooling1D, {'pool_size': 2}),
    (Flatten, {}), (Dense, dense_params(1, activation='linear')),
]

config = Config(layers=layers, cnf=cnf)
