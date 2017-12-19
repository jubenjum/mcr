"""load_segmented: helper class for cached preloading features for segmented
call recognition

"""

from __future__ import division

import warnings
from itertools import izip

import numpy as np
from numpy.lib.stride_tricks import as_strided

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from joblib import Parallel, delayed

from spectral import Spectral
from zca import ZCA
from util import wavread

import ipdb


## helper functions for the FeatureLoader class
def load_wav(fname, fs=16000):
    """ audio loader """

    sig, found_fs = wavread(fname)
    if fs != found_fs:
        raise ValueError('sampling rate should be {0}, not {1}. '
                         'please resample.'.format(fs, found_fs))
    
    if len(sig.shape) > 1:
        warnings.warn('stereo audio: merging channels')
        sig = (sig[:, 0] + sig[:, 1]) / 2
    
    return sig 


# this function goes with FeatureLoader but is defined outside it,
# because I cannot get the parallellization to work on instance methods
def extract_features_fix_stacksize(sig, noise, start, stacksize, encoder, buffer_length=0.1):

    # determine buffer and call start and end points in smp and fr
    buffer_len_smp = int(buffer_length * encoder.fs)
    buffer_len_fr = int(buffer_len_smp / encoder.fshift)

    stacksize_smp = int(stacksize * encoder.fshift)
    call_start_smp = int(start * encoder.fs) + buffer_len_smp
    call_end_smp = call_start_smp + stacksize_smp

    # the part we're gonna cut out: [buffer + call + buffer]
    slice_start_smp = call_start_smp - buffer_len_smp
    slice_end_smp = call_end_smp + buffer_len_smp

    # pad signal
    sig = np.pad(sig, (buffer_len_smp, buffer_len_smp), 'constant')
    sig_slice = sig[slice_start_smp: slice_end_smp]

    # extract features and cut out call
    feat = encoder.transform(sig_slice, noise_profile=noise)
    feat = feat[buffer_len_fr: buffer_len_fr + stacksize]

    # pad at the end
    feat = np.pad(feat, ((0, stacksize - feat.shape[0]), (0, 0)), 'constant')
    return feat


#################################
# extract features
def extract_features(sig, noise, start, end, encoder, buffer_length=0.1):

    # determine buffer and call start and end points in smp and fr
    buffer_len_smp = int(buffer_length * encoder.fs)
    buffer_len_fr = int(buffer_len_smp / encoder.fshift)

    call_start_smp = int(start * encoder.fs) + buffer_len_smp
    call_end_smp =  int(end * encoder.fs) + buffer_len_smp

    # the part we're gonna cut out: [buffer + call + buffer]
    slice_start_smp = call_start_smp - buffer_len_smp
    slice_end_smp = call_end_smp + buffer_len_smp

    # pad signal
    sig = np.pad(sig, (buffer_len_smp, buffer_len_smp), 'constant')
    sig_slice = sig[slice_start_smp: slice_end_smp]

    # extract features and cut out call
    feat = encoder.transform(sig_slice, noise_profile=noise)
    feat = feat[buffer_len_fr: -buffer_len_fr]

    # pad at the end
    #feat = np.pad(feat, ((0, stacksize - feat.shape[0]), (0, 0)), 'constant')
    return feat

def extract_noise(sig, cfg, encoder):
    
    if cfg['n_noise_fr'][0] == 0:
        noise = None
    else:
        nsamples = (cfg['n_noise_fr'][0] + 2) * encoder.fshift
        spec = encoder.get_spectrogram(sig[:nsamples])[2:, :]
        noise = spec.mean(axis=0)
        noise = np.clip(noise, 1e-4, np.inf)
        
    return noise


def encoder_func(cfg, encoder=Spectral):
    '''wrap the encoder ...'''

    # self.feat_param should have the right attributes
    # that will be used to create the encoder arguments
    encoder_attr_ = []
    for var_name, var_value in cfg.items():
        if var_name in ['normalize', 'n_noise_fr', 'stacksize']:
            continue
        if isinstance(var_value[0], unicode):
            encoder_attr_.append("{0}='{1}'".format(var_name, var_value[0]))
        else:
            encoder_attr_.append("{0}={1}".format(var_name, var_value[0]))

    # dynamically build and run the encoder with its defined attributes
    _encoder_args = ', '.join([str(w) for w in encoder_attr_])
    _encoder_comm = "_encoder = encoder({})".format(_encoder_args)
    exec(_encoder_comm)

    return _encoder 

###########################



class IdentityTransform(TransformerMixin, BaseEstimator):
    """Dummy Transformer that implements the identity transform
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class FeatureLoader(TransformerMixin, BaseEstimator):
    ''' FeatureLoader '''

    # TODO: encoder_vars should be build from a configuration file 
    # encoder_vars are the variables that the class Spectral uses
    encoder_vars = ['fs', 'window_length', 'window_shift',
            'nfft', 'lowerf', 'upperf', 'nfilt',
            'taper_filt', 'dct', 'nceps',
            'log_e', 'lifter', 'deltas', 'remove_dc',
            'medfilt_t', 'medfilt_s', 'noise_fr', 'pre_emph']

    feat_param = {'stacksize':40,
                  'normalize':'mvn',
                  'n_noise_fr':0,
                  'fs':16000,
                  'window_length':0.050,
                  'window_shift':0.010,
                  'nfft':1024,
                  'scale':'mel',
                  'lowerf':120,
                  'upperf':7000,
                  'nfilt':40,
                  'taper_filt':True,
                  'compression':'log',
                  'dct':False,
                  'nceps':13,
                  'log_e':True,
                  'lifter':22,
                  'deltas':False,
                  'remove_dc':False,
                  'medfilt_t':0,
                  'medfilt_s':(0, 0),
                  'noise_fr':0,
                  'pre_emph':0.97,
                  'n_jobs':1, 'verbose':False}

    CACHE = ['feat_cache', 'noise_cache', 'wav_cache'] 

    def __init__(self, encoder=Spectral, **kwargs):
         
        # update the class attributes and should contain 
        # the attributes of needed by the encoder
        if kwargs:
            self.feat_param.update(kwargs) # the parameters used by the encoder
        self.__dict__.update(self.feat_param)

        if self.normalize == 'mvn':
            self.normalizer = StandardScaler()
        elif self.normalize == 'zca':
            self.normalizer = ZCA()
        elif self.normalize == 'minmax':
            self.normalizer = MinMaxScaler()
        else:
            self.normalizer = IdentityTransform()
        
        # self.feat_param should have the right attributes
        # that will be used to create the encoder arguments
        encoder_attr_ = []
        for var_name, var_value in self.feat_param.items():
            if var_name in self.encoder_vars:
                if isinstance(var_value, str):
                    encoder_attr_.append("{0}='{1}'".format(var_name, var_value))
                else:
                    encoder_attr_.append("{0}={1}".format(var_name, var_value))
        
        # dynamically build and run the encoder with its defined attributes
        _encoder_args = ', '.join([str(w) for w in encoder_attr_])
        _encoder_comm = "self.encoder = encoder({})".format(_encoder_args)

        exec(_encoder_comm)

        if not hasattr(self, 'feat_cache'):
            setattr(self, 'feat_cache', {})
       
        if not hasattr(self, 'noise_cache'):       
            setattr(self, 'noise_cache', {})
 
        if not hasattr(self, 'wav_cache'):
            setattr(self, 'wav_cache', {})

        self.D = self.encoder.n_features * self.stacksize
   
    def actualize_data(self, **kwargs):
        '''actualize the data inside FeatureLoader class, including
        caches and internal class variables'''
         
        if kwargs:
            self.__dict__.update(kwargs)

    def clear_cache(self):
        self.wav_cache = {}
        self.noise_cache = {}
        self.feat_cache = {}

    def get_params(self, deep=True):
        #p = super(FeatureLoader, self).get_params()
        REMOVED_PARAM = ['n_jobs', 'verbose']
        p = {k:v for k, v in self.__dict__.items() if k not in REMOVED_PARAM}
        return p

    def get_key(self):
        """'Frozen' dictionary representation of this object's parameters.
        Used as key in caching.
        """
        p = self.get_params()
        encoder_param = tuple(sorted((k, v) for k, v in p.items() if k not in self.CACHE))
        return encoder_param 

    def _load_wav(self, fname):
        """
        Memoized audio loader.
        """
        if not fname in self.wav_cache:
            self.wav_cache[fname] = load_wav(fname, fs=self.fs)
        
        return self.wav_cache[fname]

    def _fill_noise_cache(self, X):
        for fname in X[:, 0]:
            self._extract_noise(fname)

    def _extract_noise(self, fname):
        cfg = (
            ('fs', self.fs),
            ('window_length', self.window_length),
            ('window_shift', self.window_shift),
            ('nfft', self.nfft),
            ('remove_dc', self.remove_dc),
            ('medfilt_t', self.medfilt_t),
            ('medfilt_s', self.medfilt_s),
            ('pre_emph', self.pre_emph)
        )
        key = (fname, cfg)
        
        if not key in self.noise_cache:
            if self.n_noise_fr == 0:
                self.noise_cache[key] = None
            else:
                sig = self._load_wav(fname)
                nsamples = (self.n_noise_fr + 2) * self.encoder.fshift
                spec = self.encoder.get_spectrogram(sig[:nsamples])[2:, :]
                noise = spec.mean(axis=0)
                noise = np.clip(noise, 1e-4, np.inf)
                self.noise_cache[key] = noise
        return self.noise_cache[key]

    def _fill_feat_cache(self, X_keys):
        sigs = [self._load_wav(fname) for fname, _ in X_keys]
        noises = [self._extract_noise(fname) for fname, _ in X_keys]
    
        p = []
        if self.n_jobs == 1:
            for (fname, start), sig, noise in izip(X_keys, sigs, noises):
                r = extract_features_fix_stacksize(sig, noise, start, self.stacksize, self.encoder)
                p.append(r)

        else:
            p = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(extract_features_fix_stacksize)(
                    sig, noise, start, self.stacksize, self.encoder)
                for (fname, start), sig, noise in izip(X_keys, sigs, noises))

        r = {x_key: feat for x_key, feat in izip(X_keys, p)}
        key = self.get_key()
        self.feat_cache[key].update(r)

    def get_specs(self, X):
        #p = self.get_params()
        #class_attributes = tuple(sorted(p.items())) 
        class_attributes = self.get_key()

        # list of [(filename, start)]
        X_keys = [(X[ix, 0], X[ix, 1]) for ix in xrange(X.shape[0])]
        
        # feat_cache must have the same attributes that the class
        if class_attributes in self.feat_cache:
            # check for missing keys
            missing_X_keys = [
                x_key
                for x_key in X_keys
                if not x_key in self.feat_cache[class_attributes]
            ]
            self._fill_feat_cache(missing_X_keys)
        else:
            self.feat_cache[class_attributes] = {}
            self._fill_feat_cache(X_keys)
        
        return np.vstack((self.feat_cache[class_attributes][x_key] for x_key in X_keys))

    def fit(self, X, y=None):
        """Load audio and optionally estimate mean and covar

        Parameters
        ----------
        X : ndarray with columns
            filename, start, end
        y :
        """
        r = self.get_specs(X)
        self.normalizer.fit(r)
        #return self

    def transform(self, X, y=None):
        """Load audio and perform feature extraction.

        Parameters
        ----------
        X : ndarray
        """
        r = self.get_specs(X)
        r = self.normalizer.transform(r)
        return as_strided(
            r,
            shape=(r.shape[0]//self.stacksize, r.shape[1]*self.stacksize),
            strides=(r.strides[0]*self.stacksize, r.strides[1])
        )

    def fit_transform(self, X, y=None):
        r = self.get_specs(X)
        r = self.normalizer.fit_transform(r)
        return as_strided(
            r,
            shape=(r.shape[0]//self.stacksize, r.shape[1]*self.stacksize),
            strides=(r.strides[0]*self.stacksize, r.strides[1])
        )


def split_config(kwargs):
    """ split configuration into dynamic (multiple values) and static
    (single value) dicts
    """
    dynamic = {}
    static = {}
    for k, v in kwargs.iteritems():
        if k == 'medfilt_s':
            if isinstance(v[0], list):
                dynamic[k] = map(tuple, v)
                static[k] = tuple(v[0])
            else:
                dynamic[k] = (tuple(v),)
                static[k] = tuple(v)
        else:
            if isinstance(v, list):
                dynamic[k] = v
                static[k] = v[0]
            else:
                dynamic[k] = (v,)
                static[k] = v
    return dynamic, static


def ensure_list(kwargs):
    r = {}
    for k, v in kwargs.iteritems():
        if k == 'medfilt_s':
            if isinstance(v[0], list):
                r[k] = map(tuple, v)
            else:
                r[k] = (tuple(v), )
        else:
            if isinstance(v, list):
                r[k] = v
            else:
                r[k] = (v,)
    return r

def ensure_single(kwargs):
    r = {}
    for k, v in kwargs.iteritems():
        if k == 'medfilt_s':
            if isinstance(v[0], list):
                r[k] = tuple(v[0])
            else:
                r[k] = tuple(v)
        else:
            if isinstance(v, list):
                r[k] = v[0]
            else:
                r[k] = v
    return r
