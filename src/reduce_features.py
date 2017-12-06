#!/usr/bin/env python

"""
reduce_features: reduce features from the csv files.
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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import ipdb

from mcr.util import load_config


REDUCTION_METHODS =  ['PCA', 'LDA', 'RAW', 'TSNE']
MATRIX_METHODS =  ['PCA', 'LDA', 'TSNE']


def main():
    import argparse

    parser = argparse.ArgumentParser( prog=sys.argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='prepare the csv or abx files to compute ABX score')

    parser.add_argument('features_source', help='csv file contining hte features')

    parser.add_argument('algorithm_config', help='algorithm configuration file for the feature extration')

    parser.add_argument('-o', '--out_csv', required=True, help='output features and labels in csv format')

    parser.add_argument('--standard_scaler', action='store_true', default=False, required=False,
            help='scale the features')
    
    parser.add_argument('-r', '--reduction', help=('use dimension reduction, '
        'valid methods are raw, pca, lda or tsne'))
    
    args = parser.parse_args()
    data_file = args.features_source
    reduction_type = args.reduction
    config_file = args.algorithm_config
    output_csv = args.out_csv
    standard_scaler = args.standard_scaler

     
    ###### CONFIGURATION
    config = load_config(config_file)

    if reduction_type:
        red_method = reduction_type
        red_method = red_method.upper()
        if red_method not in REDUCTION_METHODS:
            print('--reduction valid algorithms: {} -- "{}" given'.format('  '.join(REDUCTION_METHODS),
                  red_method))
            sys.exit()

        if red_method=='RAW':
            red_method = None
        else:
            try:
                new_dimension = config['dimension_reduction'][red_method]  # from the config file 
            except:
                print('missing section dimension_reduction in config file [{}]'.format(red_method))
                sys.exit()

    else: 
       red_method = None


    ###### FEATURES: where 
    ## label,feat1,feat2..featNJ where J is the  total number of labels, N number of features 
    
    #columns = ['label'] 
    #df = pd.read_csv(data_file, names=columns, engine='python')
    #X = df[['filename', 'start', 'end']].values
    #labels = df['label'].values

    ## FIX: what if the format changes?
    labels = []
    features = []
    with open(data_file, 'r') as dfile:
        for line in dfile.readlines():
            row = line.strip().split(',') # TODO 
            labels.append(row[0]) # label/call in the first column
            features.append([float(x) for x in row[1:]])


    ###### FEATURES feature reduction
    X_feat = np.array(features)
    is_matrix = False if np.object == X_feat.dtype else True
   
    if not is_matrix and (red_method in MATRIX_METHODS):
        print('all features should have the same dimension')
        sys.exit()

    if standard_scaler and is_matrix:
        X_feat = StandardScaler().fit_transform(X_feat)

    if red_method == 'PCA' and is_matrix: 
	pca = PCA(n_components=new_dimension)
	reduced_embeddings = pca.fit_transform(X_feat)

    elif red_method == 'LDA' and is_matrix:
	lda = LinearDiscriminantAnalysis(n_components=new_dimension)
	reduced_embeddings = lda.fit_transform(X_feat, labels)
    
    elif red_method == 'TSNE' and is_matrix:
        tsne = TSNE(n_components=new_dimension)
        reduced_embeddings = tsne.fit_transform(X_feat)

    else: # default = raw
        reduced_embeddings = X_feat   
        

    with open(output_csv, 'w') as emb_csv: 
        for label, feats in zip(labels, reduced_embeddings):
            t = '{},'.format(label) + ','.join(['{}'.format(x) for x in feats]) + '\n'
            emb_csv.write(t)


if __name__ == '__main__':
    main()
