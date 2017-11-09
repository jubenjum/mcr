"""util: miscellaneous helper functions
"""
from __future__ import division, print_function

from time import time
import sys
from contextlib import contextmanager
import string
from itertools import product, tee
from math import ceil, log
from functools import partial
import os.path
import ipdb

import numpy as np
import scipy.io.wavfile
import toml
import sklearn.metrics
import h5features as h5f


def make_f1_score(average):
    """Return sklearn-style scorer object which measures f1-score with the
    specified averaging method.

    Returns
    -------
    Scorer object

    """
    func = partial(sklearn.metrics.f1_score, average=average)
    func.__name__ = 'f1_score'
    func.__doc__ = sklearn.metrics.f1_score.__doc__
    return sklearn.metrics.make_scorer(func)


def resample(X_train, y_train, minority_factor=2):
    """Resample the majority class.

    Reduce the size of the majority class to `minority_factor` * the size of the
    second largest class.

    """
    y_counts = np.bincount(y_train)
    minority_class = np.argsort(-y_counts)[1]  # second biggest class
    minority_size = y_counts[minority_class]
    ixs = []
    for label in np.unique(y_train):
        ixs_for_label = np.nonzero(y_train == label)[0]
        ixs.extend(
            list(np.random.choice(
                ixs_for_label,
                min(len(ixs_for_label), minority_size * minority_factor),
                replace=False
            ))
        )
    return X_train[ixs, :], y_train[ixs]


def wavread(filename):
    """Read wave file

    Returns
    -------
    sig : ndarray
        signal
    fs : int
        samplerate
    """
    fs, sig = scipy.io.wavfile.read(filename)
    return sig, fs


@contextmanager
def verb_print(msg, verbose=False):
    """Helper for verbose printing with timing around pieces of code.
    """
    if verbose:
        t0 = time()
        msg = msg + '...'
        print(msg, end='')
        sys.stdout.flush()
    try:
        yield
    finally:
        if verbose:
            print('done. time: {0:.3f}s'.format(time() - t0))
            sys.stdout.flush()


def load_config(filename):
    """Load configuration from file.
    """
    with open(filename) as fid:
        config = toml.loads(fid.read())
    return config


def generate_abx_files(features, annotations, file_name='abx_dataw'):
    """Generate files for ABXpy:

    Input
    -----
    features   : feature cache produced by load_segmented.FeatureLoader
    annotations:
    file_name  :

    """

    item_file = '{}.item'.format(file_name)
    features_file = '{}.features'.format(file_name)

    # the first element in features is is the encoder parameters
    data_features = features[features.keys()[0]]

    all_features = []
    times = []
    files = []
    with open(item_file, 'w') as ifile:
        ifile.write("#file onset offset #call\n") 
        for n, (file_description, features_selection)   in enumerate(data_features.iteritems()):
            #ipdb.set_trace()
            audio_file, start_audio = file_description

            # selelected_annots have filename, start, end, label
            selected_annot = annotations.loc[(annotations['filename'] == audio_file) &
                    (annotations['start'] == start_audio)]
            #print('++++++++++++++++++++++++++++ '+ audio_file + ' ' + str(len(selected_annot)))
            #print(selected_annot)
            
            # data have some issues with doublets
            # I am taking the first, TODO: select other than first
            if len(selected_annot) > 1: 
                selected_annot = selected_annot.iloc[0]

            v_ = np.array([features_selection.flatten()])
            all_features.append(v_)
            times.append(float(selected_annot['start']) + 0.5*(float(selected_annot['end']) -
                         float(selected_annot['start']))
                         )
            bname = '{}_{:04d}'.format(os.path.basename(audio_file), n)
            #print('************************ '+bname)
            files.append(bname)
            ifile.write("{} {} {} {}\n".format(bname,
                float(selected_annot['start']), 
                float(selected_annot['end']),
                selected_annot['label'].values[0]))

    labels = list(np.array([times]))
    data = h5f.Data(files, labels, all_features, check=False)
    h5_features = h5f.Writer(features_file)
    h5_features.write(data, 'features')
    h5_features.close()


def pretty_cm(cm, labels, hide_zeros=False, offset=''):
    """Pretty print for confusion matrices.
    """
    valuewidth = int(np.log10(np.clip(cm, 1, np.inf)).max()) + 1
    columnwidth = max(map(len, labels)+[valuewidth]) + 1
    empty_cell = " " * columnwidth
    s = ''
    # header
    s += offset + empty_cell
    for label in labels:
        s += "{1:>{0}s}".format(columnwidth, label)
    s += '\n\n'
    # rows
    for i, label1 in enumerate(labels):
        s += offset + '{1:{0}s}'.format(columnwidth, label1)
        for j in range(len(labels)):
            cell = '{1:{0}d}'.format(columnwidth, cm[i, j])
            if hide_zeros:
                cell = cell if cm[i, j] != 0 else empty_cell
            s += cell
        s += '\n'
    return s


def string_to_bool(s):
    """yeah.
    """
    if s == 'True':
        return True
    elif s == 'False':
        return False
    raise ValueError('not parsable')


def roll_array(arr, stacksize):
    arr = np.vstack((
        np.zeros((stacksize//2, arr.shape[1])),
        arr,
        np.zeros((stacksize//2, arr.shape[1]))
    ))
    return np.hstack(
        np.roll(arr, -i, 0)
        for i in range(stacksize)
    )[:arr.shape[0] - stacksize + 1]


def encode_symbol_range(high,
                        symbols=string.ascii_lowercase,
                        join=lambda s: ''.join(s)):
    return dict(
        enumerate(
            map(
                join,
                product(
                    *tee(symbols,
                         int(ceil(log(high, len(symbols)))))
                )
            )
        )
    )
