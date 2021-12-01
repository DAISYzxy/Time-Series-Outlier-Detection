# Anomaly score names
# from _typeshed import OpenTextModeWriting
import numpy as np
import pandas as pd
from stumpy import stump

small_quantile = 0.1

# 'orig_p2p_inv','acc_std_inv',

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


    # Smooth and mask anomaly score
    padding_length = 3
    padding = w * padding_length
    seq['mask'] = 0.0
    seq.loc[seq.index[w:-w - padding], 'mask'] = 1.0
    seq['mask'] = seq['mask'].rolling(padding, min_periods=1).sum() / padding
    for name in names:
        seq[f'{name}_score'] = seq[name].rolling(w).mean() * seq['mask']

    return seq


# Parameter setting
min_window_size = 50
max_window_size = 800
growth_rate = 1.2
train_length = 10

# Determine window sizes
size = int(np.log(max_window_size / min_window_size) / np.log(growth_rate)) + 1
rates = np.full(size, growth_rate) ** np.arange(size)
ws = (min_window_size * rates).astype(int)

# Path setting
import pathlib
import tqdm

rootpath = pathlib.Path('../dataset')
txt_dirpath = rootpath / 'phase2'  # Place the txt files in this directory

# Evaluate anomaly score for each time series
results = []
for txt_filepath in sorted(txt_dirpath.iterdir()):
    if ".DS_Store" in str(txt_filepath): continue
    # Load time series
    X = np.loadtxt(txt_filepath)
    number = txt_filepath.stem.split('_')[0]
    split = int(txt_filepath.stem.split('_')[-1])
    print(f'\n{txt_filepath.name} {split}/{len(X)}', flush=True)

    # Evaluate anomaly score for each window size w
    for w in tqdm.tqdm(ws):

        # Skip long subsequence
        if w * train_length > split:
            continue

        # Compute anomaly score
        seq = run(X, w, split)

        # Evaluate anomaly score
        for name in names:

            # Copy anomaly score
            y = seq[f'{name}_score'].copy()

            # Find local maxima
            cond = (y == y.rolling(w, center=True, min_periods=1).max())
            y.loc[~cond] = np.nan

            # Find 1st peak
            index1 = y.idxmax()
            value1 = y.max()

            # Skip if all score is NaN
            if not np.isfinite(value1):
                continue

            # Skip if all score is not positive
            if value1 <= 0.0:
                continue

            # Skip if train data has 1st peak
            begin = index1 - w
            end = index1 + w
            if begin < split:
                continue

            # Find 2nd peak
            y.iloc[begin:end] = np.nan
            index2 = y.idxmax()
            value2 = y.max()

            # Skip if 2nd peak height is zero
            if value2 == 0:
                continue

            # Evaluate rate of 1st peak height to 2nd peak height
            rate = value1 / value2
            results.append([number, w, name, rate, begin, end, index1, value1, index2, value2])

# Display results
results = pd.DataFrame(results, columns=['number', 'w', 'name', 'rate', 'begin', 'end', 'index1', 'value1', 'index2', 'value2'])
submission = results.loc[results.groupby('number')['rate'].idxmax(), ['number','w', 'name', 'rate', 'begin', 'end', 'index1', 'value1', 'index2', 'value2']]
submission.index.number = 'No.'
submission.to_csv('result.csv')

