"""
util: miscellaneous helper functions
"""
from __future__ import division, print_function

import random
import string
import sys
import os.path
import os

from time import time
from contextlib import contextmanager
from itertools import product, tee
from itertools import izip_longest, cycle
from math import ceil, log
from functools import partial

from joblib import Memory
import numpy as np
import h5py

import scipy.io.wavfile
import toml
import sklearn.metrics

# for my_LinearDiscriminantAnalysis
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

# for AutoEncoder and LSTM
stderr = sys.stderr  # avoid kears messages
sys.stderr = open(os.devnull, 'w')

from keras import regularizers
from keras.layers import Input, Dense, Masking
from keras.layers import LSTM, RepeatVector
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import load_model

from keras import losses
sys.stderr = stderr


# from https://docs.python.org/2/library/itertools.html#itertools.tee
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


# Module initialization
np.random.seed(42)


# creating a global memory for the package in the directory
# where the scripts are running
def get_cache_dir():
    ''' get the directory where the cache is stored, by default it will create a
    directory ".cache" in the current directory
    '''
    cdir = os.curdir+'/.cache'
    if not os.path.exists(cdir):
        os.makedirs(cdir)
    return cdir


def build_cache():
    return Memory(cachedir=get_cache_dir(), verbose=0)


memory = build_cache()


def normalize(features):
    ''' Normalize features in the range (0,1) '''
    x = np.ma.array(features, mask=np.isnan(features))
    x_ = (x - x.mean(axis=0)) / x.std(axis=0)

    # other options that didn't give good results:
    # x_ = (x - x.min()) / (x.max() - x.min()) + np.finfo(float).eps
    # x_ = (x - x.min()) / (x.max() - x.min())
    
    x_norm = x_.data
    x_norm[x_norm == 0.0] += np.finfo(float).eps
    x_norm[x.mask] = 0.0
    return x_norm


# LSTM Encoder https://blog.keras.io/building-autoencoders-in-keras.html
class KR_LSMTEncoder:
    def __init__(self, features, labels, input_dim=40):
        self.num_feat, self.feat_dim = features.shape
        self.timesteps = self.feat_dim // input_dim  # frames
        self.input_dim = input_dim  # size of the features/nfilt
        self.labels = labels

        # normalize and set nan as a mask
        self.features = normalize(features)
        self.features = np.array([x.reshape(len(x)//self.input_dim,
                                            self.input_dim)
                                  for x in self.features])

        # mixing the order of the features ...
        random.seed(42)
        shuffled_range = range(self.num_feat)
        random.shuffle(shuffled_range)
        self.features_ = self.features[shuffled_range]
        self.training_data = self.features_

        # split: train (85%), validation(15%) and test (0%)
        # p85 = int(len(self.features)*0.85)
        # self.training_data = self.features_[:p85, :, :]
        # self.validation_data = self.features_[p85:, :, :]
        # self.trn_generator = self._generator(self.training_data, 40)
        # self.val_generator = self._generator(self.validation_data, 40)

    def _generator(self, data, n_batches=40):
        for x in grouper(cycle(data), n_batches):
            yield np.array(x), np.array(x)

    def get_model(self, n_dimensions=40):
        inputs = Input(shape=(self.timesteps, self.input_dim))
        mask = Masking(mask_value=0.0)(inputs)
        encoded = LSTM(n_dimensions, return_sequences=False)(mask)
        decoded = RepeatVector(self.timesteps)(encoded)
        decoded = LSTM(self.input_dim, return_sequences=True)(decoded)

        self.autoencoder = Model(inputs, decoded)
        self.encoder = Model(inputs, encoded)

    def save_data(self, filename):
        """save the features and labels in a hdf5"""
        with h5py.File (filename) as f:
            f['features'] = self.features
            f['labels'] = self.labels

    def save_encoder(self, file_name):
        """ save the encoder LSTM in h5 format using keras save function, to
        get the model back use

        >> model = load_model('my_model.h5')

        """
        self.encoder.save("{}".format(file_name))

    def load_encoder(self, file_name):
        """ Get the autoencoder saved with save_encoder """
        self.encoder = load_model(file_name)

    def fit(self, n_dimensions):
        self.get_model(n_dimensions)
        self.autoencoder.compile(optimizer='rmsprop',
                                 loss='mse',
                                 metrics=['acc'])

        #self.history = self.autoencoder.fit_generator(
        #    self.trn_generator, steps_per_epoch=1000, epochs=1000,
        #    validation_data=self.val_generator, validation_steps=40)

        self.history = self.autoencoder.fit(self.training_data,
                                            self.training_data, epochs=100)

        self.history_dict = self.history.history
        loss_values = self.history_dict['loss']
        acc_values = self.history_dict['acc']


    def reduce(self, *new_features):
        if new_features:
            return self.encoder.predict(new_features[0])
        else:
            return self.encoder.predict(self.features)


# autoencode https://blog.keras.io/building-autoencoders-in-keras.html
class KR_AutoEncoder:
    def __init__(self, features, labels):
        # self.features = normalize(features)
        self.features = features
        self.labels = labels
        self.num_feat, self.feat_dim = features.shape

    def fit(self, n_dimensions):
        self.n_dimensions = n_dimensions
        input_call = Input(shape=(self.feat_dim,))

        encoded = Dense(n_dimensions, activation='sigmoid',
                        activity_regularizer=regularizers.l1(10e-5))(input_call)
        decoded = Dense(self.feat_dim)(encoded)

        # # DEFINE THE ENCODER LAYERS
        # encoded = Dense(n_dimensions*4, activation = 'relu')(input_call)
        # encoded = Dense(n_dimensions, activation = 'relu')(encoded)

        # # DEFINE THE DECODER LAYERS
        # decoded = Dense(n_dimensions*4, activation = 'relu')(encoded)
        # decoded = Dense(self.feat_dim, activation = 'sigmoid')(decoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_call, decoded)

        # this model maps an input to its encoded representation
        self.encoder = Model(input_call, encoded)

        epochs = 1000
        #self.autoencoder.compile(optimizer='adadelta', loss='mse')
        self.autoencoder.compile(optimizer='rmsprop', loss='mse')

        stop_callback = [EarlyStopping(monitor='val_loss',
                                       patience=epochs//10, verbose=0), ]
        self.autoencoder.fit(self.features, self.features,
                             shuffle=False,
                             epochs=epochs,
                             callbacks=stop_callback,
                             validation_data=(self.features, self.features))

    def decode(self):
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.n_dimensions,))

        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]

        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))


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
            warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
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
    fs, sig = scipy.io.wavfile.read(os.path.expanduser(filename))
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
    """ convert the strings 'True' in True and 'False' in False

    >>> string_to_bool('True')
    True

    >>> string_to_bool('False')
    False

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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
