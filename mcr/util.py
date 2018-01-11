"""
util: miscellaneous helper functions
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

import numpy as np
import scipy.io.wavfile
import toml
import sklearn.metrics

# for my_LinearDiscriminantAnalysis
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping

# encoding a LSTM
class KR_LSMTEncoder:
    ''' from '''
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.num_feat, self.feat_dim = features.shape

    def fit(self, n_dimensions):
        pass

    def reduce(self):
        pass


# autoencoder using keras
class KR_AutoEncoder:
    ''' from https://blog.keras.io/building-autoencoders-in-keras.html '''
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.num_feat, self.feat_dim = features.shape

    def fit(self, n_dimensions):
        encoding_dim = n_dimensions
        input_call = Input(shape=(self.feat_dim,))

        encoded = Dense(encoding_dim, activation='sigmoid',
                        activity_regularizer=regularizers.l1(10e-5))(input_call)
        decoded = Dense(self.feat_dim)(encoded)

        #### DEFINE THE ENCODER LAYERS
        ###encoded1 = Dense(encoding_dim*4, activation = 'relu')(input_call)
        ###encoded2 = Dense(encoding_dim*3, activation = 'relu')(encoded1)
        ###encoded3 = Dense(encoding_dim*2, activation = 'relu')(encoded2)
        ###encoded = Dense(encoding_dim, activation = 'relu')(encoded3)

        #### DEFINE THE DECODER LAYERS
        ###decoded1 = Dense(encoding_dim*2, activation = 'relu')(encoded)
        ###decoded2 = Dense(encoding_dim*3, activation = 'relu')(decoded1)
        ###decoded3 = Dense(encoding_dim*4, activation = 'relu')(decoded2)
        ###decoded = Dense(self.feat_dim, activation = 'sigmoid')(decoded3)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_call, decoded)

        # this model maps an input to its encoded representation
        self.encoder = Model(input_call, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))

	## retrieve the last layer of the autoencoder model
        #decoder_layer = self.autoencoder.layers[-1]
        ## create the decoder model
        #decoder = Model(encoded_input, decoder_layer(encoded_input))

        epochs=1000
        callbacks = [EarlyStopping(monitor='val_loss', patience=epochs//10, verbose=0),]
        self.autoencoder.compile(optimizer='adadelta', loss='mse')
        #self.autoencoder.compile(optimizer='rmsprop', loss='mse')
        self.autoencoder.fit(self.features, self.features,
                        shuffle=True, epochs=epochs, callbacks=callbacks,
                        validation_data=(self.features, self.features))

    def reduce(self):
        return self.encoder.predict(self.features)


class my_LinearDiscriminantAnalysis(LinearDiscriminantAnalysis):

    def fit(self, X, y):
        """Fit LinearDiscriminantAnalysis model according to the given
        training data and parameters.
        .. versionchanged:: 0.19
            *store_covariance* has been moved to main constructor.
        .. versionchanged:: 0.19
            *tol* has been moved to main constructor.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array, shape (n_samples,)
            Target values.
        """
        X, y = check_X_y(X, y, ensure_min_samples=2, estimator=self)
        self.classes_ = unique_labels(y)

        if self.priors is None:  # estimate priors from sample
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            self.priors_ = np.bincount(y_t) / float(len(y))
        else:
            self.priors_ = np.asarray(self.priors)

        if (self.priors_ < 0).any():
            raise ValueError("priors must be non-negative")
        if self.priors_.sum() != 1:
            warnings.warn("The priors do not sum to 1. Renormalizing",
                        UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        # Get the maximum number of components
        if self.n_components is None:
            self._max_components = len(self.classes_) - 1
        else:
            self._max_components = max(len(self.classes_) - 1,
                                    self.n_components)

        if self.solver == 'svd':
            if self.shrinkage is not None:
                raise NotImplementedError('shrinkage not supported')
            self._solve_svd(X, y)
        elif self.solver == 'lsqr':
            self._solve_lsqr(X, y, shrinkage=self.shrinkage)
        elif self.solver == 'eigen':
            self._solve_eigen(X, y, shrinkage=self.shrinkage)
        else:
            raise ValueError("unknown solver {} (valid solvers are 'svd', "
                            "'lsqr', and 'eigen').".format(self.solver))
        if self.classes_.size == 2:  # treat binary case as a special case
            self.coef_ = np.array(self.coef_[1, :] - self.coef_[0, :], ndmin=2)
            self.intercept_ = np.array(self.intercept_[1] - self.intercept_[0],
                                    ndmin=1)
        return self


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
