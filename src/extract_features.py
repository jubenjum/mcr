#!/usr/bin/env python

"""
extract_features: prepare the csv files with features and labels used to compute ABX score.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import operator
import sys

import pandas as pd
import numpy as np
np.seterr(all='raise')
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.preprocessing import StandardScaler

from mcr.util import load_config
from mcr.util import verb_print
from mcr.util import generate_abx_files
import mcr.load_segmented

REDUCTION_METHODS =  ['PCA', 'LDA', 'RAW']


def main():
    import argparse

    parser = argparse.ArgumentParser( prog=sys.argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='prepare the csv or abx files to compute ABX score')

    parser.add_argument('stimuli_source', help='file with the stimuli source: wav_file, interval, label ')

    parser.add_argument('algorithm_config', help='algorithm configuration file for the feature extration')

    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                        default=False, help='talk more')

    parser.add_argument('-o', '--out_csv', help='output features and labels in csv format')
    
    parser.add_argument('-a', '--out_abx', required=False,
            help='output features and labels item/features abx format')

    parser.add_argument('-r', '--reduction', help=('use dimension reduction, '
        'valid methods are "raw", "pca" or "lda"'))
    
    args = parser.parse_args()

    data_file = args.stimuli_source
    config_file = args.algorithm_config
    abx_files = args.out_abx if args.out_abx else None
    verbose = args.verbose

    if args.reduction:
       red_method = args.reduction
       red_method = red_method.upper()
       if red_method not in REDUCTION_METHODS:
           print('--reduction valid algorithms: "raw", "pca" or "lda", "{}" given'.format(red_method))
           sys.exit()
       red_method = None if red_method=='RAW' else red_method
    else: 
       red_method = None
   
    ###### ANNOTATIONS
    with verb_print('reading stimuli from {}'.format(data_file),
                    verbose=verbose):
        df = pd.read_csv(data_file)
        X = df[['filename', 'start', 'end']].values
        labels = df['label'].values
        

    with verb_print('loading algorithm configuration from {}'.format(config_file),
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
            fl.actualize_data(wav_cache=wav_cache, **params)
            fl._fill_noise_cache(X)
            noise_cache.update(fl.noise_cache)
            fl.get_specs(X)
            feat_cache.update(fl.feat_cache)

    ### transforming the embeddings 
    desc_features, data_features = feat_cache.items()[0] # will remains only 1 element? 
    all_features = list()
    all_file_descriptions = list()
    all_calls = list()
    for file_description, features_selection in data_features.iteritems():
        # the calls in the same order that the embeddings
	filename, start_time = file_description
        all_calls.append(df.label.loc[(df['filename'] == filename) &
                 (df['start'] == start_time)].values[0])

        # building the embedding 
        v_ = np.array(features_selection.flatten())
        all_features.append(v_)
        all_file_descriptions.append(file_description)

    X_feat = np.array(all_features)
    #X_std = StandardScaler().fit_transform(X_feat)
    if red_method == 'PCA': 
	pca = PCA(n_components=20)
	small_embeddings = pca.fit_transform(X_feat)

    elif red_method == 'LDA':
	lda = LinearDiscriminantAnalysis(n_components=20)
	small_embeddings = lda.fit_transform(X_feat, all_calls)
    
    else: # default = raw
        small_embeddings = X_feat   
        

    # re-build the dictionary used on generate_abx_files &or save
    # files to csv file 
    if args.out_csv:
        emb_csv = open(args.out_csv, 'w') 

    new_features = dict()
    for n, file_description in enumerate(all_file_descriptions):
        new_features[file_description] = small_embeddings[n,:] 
        if args.out_csv:
            t = '"{}",'.format(all_calls[n]) + \
                 ','.join(['{}'.format(x) for x in small_embeddings[n,:]]) + '\n'
            emb_csv.write(t)

    new_feat_cache = {desc_features : new_features}

    ####### BUILD ABX files
    if abx_files:
        generate_abx_files(new_feat_cache, df, file_name=abx_files)


if __name__ == '__main__':
    main()

