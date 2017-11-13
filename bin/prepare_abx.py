#!/usr/bin/env python

"""prepare_abx: prepare the abx files to compute ABX score.

"""




import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import operator

import pandas as pd
import numpy as np
np.seterr(all='raise')
from sklearn.grid_search import ParameterGrid
#import ipdb

from mcr.util import load_config
from mcr.util import verb_print
from mcr.util import generate_abx_files
import mcr.load_segmented


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(
            prog='prepare_abx.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='prepare the abx files to compute ABX score')

        parser.add_argument('datafile', metavar='DATAFILE',
                            nargs=1,
                            help='file with training stimuli')

        parser.add_argument('config', metavar='CONFIG',
                            nargs=1,
                            help='configuration file')

        parser.add_argument('output', metavar='OUTPUT',      
                            nargs=1,
                            help='output file name')

        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')
        return vars(parser.parse_args())

    args = parse_args()

    data_file = args['datafile'][0]
    config_file = args['config'][0]
    output_file = args['output'][0]
    verbose = args['verbose']

    ###### ANNOTATIONS
    with verb_print('reading stimuli from {}'.format(data_file),
                    verbose=verbose):
        df = pd.read_csv(data_file)
        X = df[['filename', 'start', 'end']].values
        labels = df['label'].values
        #label2ix = {k: i for i, k in enumerate(np.unique(labels))}
        #y = np.array([label2ix[label] for label in labels])

    with verb_print('loading configuration from {}'.format(config_file),
                    verbose=verbose):
        config = load_config(config_file)

        features_params = mcr.load_segmented.ensure_list(config['features'])
        clf_params = mcr.load_segmented.ensure_list(config['svm'])

        CLASS_WEIGHT = clf_params['class_weight'][0]
        if not isinstance(CLASS_WEIGHT, bool):
            raise ValueError(
                'invalid value for class_weight: {}'.format(CLASS_WEIGHT)
            )
        del clf_params['class_weight']

        APPROXIMATE = clf_params['approximate'][0]
        if not isinstance(APPROXIMATE, bool):
            raise ValueError(
                'invalid value for approximation: {}'.format(APPROXIMATE)
            )
        del clf_params['approximate']


        param_grid = {}
        for k, v in features_params.iteritems():
            param_grid['features__{}'.format(k)] = v
        for k, v in clf_params.iteritems():
            param_grid['clf__{}'.format(k)] = v

    ###### FEATURES
    with verb_print('preloading audio', verbose=verbose):
        n_iter = reduce(operator.mul, map(len, features_params.values()))
        fl = mcr.load_segmented.FeatureLoader()

        wav_cache = {}
        feat_cache = {}
        noise_cache = {}

        # wav files must be sampled at 16000Hz
        for fname in X[:, 0]:
            wav_cache[fname] = mcr.load_segmented.load_wav(fname)

        for ix, params in enumerate(ParameterGrid(features_params)):
            # print 'combination {}/{}'.format(ix, n_iter)
            fl.actualize_data(wav_cache=wav_cache, **params)
            fl._fill_noise_cache(X)
            noise_cache.update(fl.noise_cache)
            fl.get_specs(X)
            feat_cache.update(fl.feat_cache)

    ###### BUILD ABX files
    #ipdb.set_trace()
    generate_abx_files(feat_cache, df, file_name=output_file)
