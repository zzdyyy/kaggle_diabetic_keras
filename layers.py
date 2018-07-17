"""prepare some layers and their default parameters for configs/*.py"""

import keras
from keras.layers import *
from keras import backend as K
from keras.regularizers import l1_l2, l2

leaky_rectify_alpha = 0.01
kernel_init_gain = 1.0
bias_init_value = 0.05
regular_factor_l1 = 0.
regular_factor_l2 = 5e-4  # weight_decay

def conv_params(filters, **kwargs):
    """default Conv2d arguments"""
    args = {
        'filters': filters,
        'kernel_size': (3, 3),
        'padding': 'same',
        'activation': lambda x: keras.activations.relu(x, leaky_rectify_alpha),
        'use_bias': True,
        'kernel_initializer': keras.initializers.Orthogonal(gain=kernel_init_gain),
        'bias_initializer': keras.initializers.constant(bias_init_value),
        'kernel_regularizer': keras.regularizers.l2(regular_factor_l2),
        'bias_regularizer': keras.regularizers.l2(regular_factor_l2),
    }
    args.update(kwargs)
    return args


def pool_params(**kwargs):
    """default MaxPool2d/RMSPoolLayer arguments"""
    args = {
        'pool_size': 3,
        'strides': (2, 2),
    }
    args.update(kwargs)
    return args


def dense_params(num_units, **kwargs):
    """default dense layer arguments"""
    args = {
        'units': num_units,
        'activation': lambda x: keras.activations.relu(x, leaky_rectify_alpha),
        'kernel_initializer': keras.initializers.Orthogonal(gain=kernel_init_gain),
        'bias_initializer': keras.initializers.constant(bias_init_value),
        'kernel_regularizer': keras.regularizers.l2(regular_factor_l2),
        'bias_regularizer': keras.regularizers.l2(regular_factor_l2),
    }
    args.update(kwargs)
    return args


class RMSPoolLayer(keras.layers.pooling._Pooling2D):
    """Use RMS(Root Mean Squared) as pooling function.

        origin version from https://github.com/benanne/kaggle-ndsb/blob/master/tmp_dnn.py
    """
    def __init__(self, *args, **kwargs):
        super(RMSPoolLayer, self).__init__(*args, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool2d(K.square(inputs), pool_size, strides,
                          padding, data_format, pool_mode='avg')
        return K.sqrt(output + K.epsilon())


class Padding2D(keras.engine.Layer):
    """layer wrapper for K.spatial_2d_padding"""

    def __init__(self, padding=((1,1), (1,1)), data_format="channels_last", **kwargs):
        super(Padding2D, self).__init__(**kwargs)
        if isinstance(padding, int):
            padding = ((padding, padding), (padding, padding))
        self.padding = padding
        self.data_format = data_format

    def call(self, x, **kwargs):
        output = K.spatial_2d_padding(x, padding=self.padding, data_format=self.data_format)
        return output

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows += sum(self.padding[0])
        cols += sum(self.padding[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])

    def get_config(self):
        config = {'padding': self.padding,
                  'data_format': self.data_format}
        base_config = super(Padding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))