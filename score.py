# Anomaly score names
from _typeshed import OpenTextModeWriting
import numpy as np
import pandas as pd
from stumpy import stump

small_quantile = 0.1

names = [
    'orig_p2p',
    'diff_p2p',
    'acc_p2p',
    'orig_p2p_inv',
    'diff_small',
    'diff_large',
    'diff_cross',
    'turkey_test_min',
    'turkey_test_max',
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

def orig_p2p_inv(seq, w):
    numer = seq['orig_p2p'].mean()
    seq['orig_p2p_inv'] = numer / seq['orig_p2p']
    return seq

def diff_small(seq, w):
    diff_abs = seq['diff'].abs()
    cond = diff_abs <= diff_abs.quantile(small_quantile)
    seq['diff_small'] = cond.rolling(w).mean().shift(-w)
    return seq

def diff_large(seq, w):
    diff_abs = seq['diff'].abs()
    cond = diff_abs > diff_abs.quantile(small_quantile)
    seq['diff_large'] = cond.rolling(w).mean().shift(-w)
    return seq

def turkey_test_min(seq, w):
    q1 = seq['diff'].rolling(w).quantile(0.25)
    q3 = seq['diff'].rolling(w).quantile(0.75)
    min = seq['diff'].rolling(w).min()
    seq['turkey_test_min'] = ((q1 - min) / (q3 - q1)).shift(-w)
    return seq

def turkey_test_max(seq, w):
    q1 = seq['diff'].rolling(w).quantile(0.25)
    q3 = seq['diff'].rolling(w).quantile(0.75)
    max = seq['diff'].rolling(w).max()
    seq['turkey_test_max'] = ((max - q3) / (q3 - q1)).shift(-w)
    return seq

def diff_large(seq, w):
    diff_abs = seq['diff'].abs()
    cond = diff_abs > diff_abs.quantile(small_quantile)
    seq['diff_large'] = cond.rolling(w).mean().shift(-w)
    return seq


def diff_cross(seq, w):
    cond = seq['diff'] * seq['diff'].shift(1) < 0
    seq['diff_cross'] = cond.rolling(w).mean().shift(-w)
    return seq


def acc_std(seq, w):
    seq['acc_std'] = seq['acc'].rolling(w).std().shift(-w)
    return seq

def acc_std_inv(seq, w):
    numer = seq['acc_std'].mean()
    seq['acc_std_inv'] = numer / seq['acc_std']
    return seq

def mp_novelty(seq, w, split):
    mp_train = stump(seq['orig'][:split], w)
    mp_join = stump(seq['orig'][split:], w, seq['orig'][:split], ignore_trivial=False)

    mpvalue = mp_join[:, 0].astype(float)
    seq.loc[split:split + len(mpvalue) - 1, 'orig_mp_novelty'] = mpvalue  # mp

    nomindex = mp_join[:, 1].astype(int)
    nomvalue = mp_train[:, 0][nomindex].astype(float)
    seq.loc[split:split + len(mpvalue) - 1, 'orig_np_novelty'] = mpvalue / nomvalue  # norm mp

    return seq


def mp_outlier(seq, w):
    mp_all = stump(seq['orig'], w)

    mpvalue = mp_all[:, 0].astype(float)
    seq.loc[0:len(mpvalue) - 1, 'orig_mp_outlier'] = mpvalue  # mp

    nomindex = mp_all[:, 1].astype(int)
    nomvalue = mp_all[:, 0][nomindex].astype(float)
    seq.loc[0:len(mpvalue) - 1, 'orig_np_outlier'] = mpvalue / nomvalue  # norm mp

    return seq


def run(X, w, split):
    seq = pd.DataFrame(X, columns=['orig'])
    seq = orig_p2p(seq, w)
    seq = diff_p2p(seq, w)
    seq = acc_p2p(seq, w)
    seq = orig_p2p_inv(seq, w)
    seq = acc_std(seq, w)
    seq = acc_std_inv(seq, w)
    seq = diff_small(seq, w)
    seq = diff_large(seq, w)
    seq = turkey_test_min(seq,w)
    seq = turkey_test_max(seq, w)
    seq = diff_cross(seq, w)
    seq = mp_novelty(seq, w, split)
    seq = mp_outlier(seq, w)

    return seq
