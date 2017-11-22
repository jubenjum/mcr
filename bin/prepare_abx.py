#!/usr/bin/env python

"""
prepare_abx: prepare the csv files with features and labels used to compute ABX score.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import operator
import sys

import pandas as pd
import numpy as np
np.seterr(all='raise')
from sklearn.grid_search import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.lda import LDA
#from sklearn.preprocessing import StandardScaler

from mcr.util import load_config
from mcr.util import verb_print
from mcr.util import generate_abx_files
import mcr.load_segmented


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        prog='prepare_abx.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='prepare the abx files to compute ABX score')

    parser.add_argument('datafile', metavar='DATAFILE',
                        help='file with training stimuli')

    parser.add_argument('config', metavar='CONFIG',
                        help='configuration file')

    parser.add_argument('output', metavar='OUTPUT',  
                        help='output file name')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        dest='verbose',
                        default=False,
                        help='talk more')

    parser.add_argument('--out_csv', help='output features and labels in csv format')
    parser.add_argument('--reduction', help='use dimension reduction, valid methods are "pca" and "lda" ')
    
    args = parser.parse_args()

    data_file = args.datafile
    output_file = args.output     
    config_file = args.config
    verbose = args.verbose
    if args.reduction:
       red_method = args.reduction
       red_method = red_method.upper()
       if red_method not in ['PCA', 'LDA']:
           print('--reduction shoule be "pca" or "lda", "{}" given'.format(args.reduction))
           sys.exit()
    else:
       red_method = None
   
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

    ### reducing the size of embeddings 
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
	lda = LDA(n_components=20)
	small_embeddings = lda.fit_transform(X_feat, all_calls)
    
    else:
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
    #generate_abx_files(feat_cache, df, file_name=output_file)
    generate_abx_files(new_feat_cache, df, file_name=output_file)
