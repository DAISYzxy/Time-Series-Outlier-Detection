# Anomaly score names
from _typeshed import OpenTextModeWriting
import numpy as np
import pandas as pd


names = [
    'orig_p2p',
    'diff_p2p',
    'acc_p2p',
    'orig_p2p_inv',
    'diff_small',
    'diff_large',
    'diff_cross',
    'acc_std',
    'acc_std_inv',
    'orig_mp_novelty',
    'orig_np_novelty',
    'orig_mp_outlier',
    'orig_np_outlier',
]


def orig_p2p(seq, w):
    rolling_max = seq['orig'].rolling(w).max()
    rolling_min = seq['orig'].rolling(w).min()
    seq['orig_p2p'] = (rolling_max - rolling_min).shift(-w)
    return seq


def diff_p2p(seq, w):
    seq['diff'] = seq['orig'].diff(1)
    rolling_max = seq['diff'].rolling(w).max()
    rolling_min = seq['diff'].rolling(w).min()
    seq['diff_p2p'] = (rolling_max - rolling_min).shift(-w)
    return seq


def acc_p2p(seq, w):
    seq['acc'] = seq['diff'].diff(1)
    rolling_max = seq['acc'].rolling(w).max()
    rolling_min = seq['acc'].rolling(w).min()
    seq['acc_p2p'] = (rolling_max - rolling_min).shift(-w)
    return seq


def acc_std(seq, w):
    seq['acc_std'] = seq['acc'].rolling(w).std().shift(-w)
    return seq




def run(X, w):
    seq = pd.DataFrame(X, columns=['orig'])
    seq = orig_p2p(seq, w)
    seq = diff_p2p(seq, w)
    seq = acc_p2p(seq, w)
    seq = acc_std(seq, w)
    return