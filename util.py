from datetime import datetime
import importlib
import subprocess
import os

import numpy as np

from quadratic_weighted_kappa import quadratic_weighted_kappa


def float32(k):
    return np.cast['float32'](k)


def kappa(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = y_true.dot(list(range(y_true.shape[1])))
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred.dot(list(range(y_pred.shape[1])))
    try:
        return np.float32(quadratic_weighted_kappa(y_true, y_pred))
    except IndexError:
        return np.float32(np.nan)
    except ZeroDivisionError:
        return np.float32(np.nan)


def kappa_from_proba(w, p, y_true):
    return kappa(y_true, p.dot(w))


def load_module(mod):
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def get_commit_sha():
    p = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                         stdout=subprocess.PIPE)
    output, _ = p.communicate()
    return output.strip().decode('utf-8')


def get_submission_filename():
    sha = get_commit_sha()
    return "data/sub_{}_{}.csv".format(sha,
                                       datetime.now().replace(microsecond=0))

