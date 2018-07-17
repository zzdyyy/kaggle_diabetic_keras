import pprint
import os

import numpy as np

from util import mkdir
from data import FEATURE_DIR

mkdir(FEATURE_DIR)

class Config(object):
    def __init__(self, layers, cnf=None):
        self.layers = layers
        self.cnf = cnf
        pprint.pprint(cnf)

    def get(self, k, default=None):
        return self.cnf.get(k, default)

    @property
    def weights_epoch(self):
        path = "weights/{}/epochs".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, '{epoch:03d}_{val_loss:.3e}_{kappa:.3f}.h5')

    @property
    def weights_best(self):
        path = "weights/{}/best".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, '{epoch:03d}_{val_loss:.3e}_{kappa:.3f}.h5')

    @property
    def weights_file(self):
        path = "weights/{}".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, 'weights.h5')

    @property
    def retrain_weights_file(self):
        path = "weights/{}/retrain".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, 'weights.h5')

    @property
    def final_weights_file(self):
        path = "weights/{}".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, 'weights_final.h5')

    def get_features_fname(self, n_iter, skip=0, test=False):
        fname = '{}_{}_mean_iter_{}_skip_{}.npy'.format(
            self.cnf['name'], ('test' if test else 'train'),  n_iter, skip)
        return os.path.join(FEATURE_DIR, fname)

    def get_std_fname(self, n_iter, skip=0, test=False):
        fname = '{}_{}_std_iter_{}_skip_{}.npy'.format(
            self.cnf['name'], ('test' if test else 'train'), n_iter, skip)
        return os.path.join(FEATURE_DIR, fname)

    def save_features(self, X, n_iter, skip=0, test=False):
        np.save(open(self.get_features_fname(n_iter, skip=skip,
                                              test=test), 'wb'), X)

    def save_std(self, X, n_iter, skip=0, test=False):
        np.save(open(self.get_std_fname(n_iter, skip=skip,
                                        test=test), 'wb'), X)

    def load_features(self, test=False):
        return np.load(open(self.get_features_fname(test=test)))

