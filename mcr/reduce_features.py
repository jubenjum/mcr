#!/usr/bin/env python

"""
reduce_features: reduce features from the csv files.
"""

import warnings
import sys

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from mcr.util import load_config
from mcr.util import KR_AutoEncoder
from mcr.util import KR_LSMTEncoder
from mcr.util import my_LinearDiscriminantAnalysis
from mcr.util import build_cache


# Modules configuration
warnings.filterwarnings("ignore")
np.seterr(all='raise')
np.seterr(under="ignore")


memory = build_cache()

__all__ = ['dimension_reduction']

REDUCTION_METHODS = ['PCA', 'LDA', 'RAW', 'TSNE', 'AE', 'LSA', 'LSTM']
MATRIX_METHODS = ['PCA', 'LDA', 'TSNE']


@memory.cache
def dimension_reduction(features, labels, red_method, new_dimension,
                        standard_scaler=False, config=None):
    ''' dimesion_deduction is a wrap to PCA, LDA, TSNE, LSA dimension
    reduction methods

    The input features will be reduced with one of these methods (or raw
    output), it can be removed the std from the features using the std.

    Parameters
    ----------
    features: list of numpy arrays with numeric floating point elements
              (list[numpy.ndarray])
    labels: values of the same size of features (list)
    red_method: the reduction method, valid methods are:
                'PCA', 'LDA', 'RAW', 'TSNE', 'AE', 'LSH' (str)
    new_dimension: new dimension [int]
    standard_scaler: scale all features [bool]


    Returns
    -------
    shrinked_features: reduced dimension features with the same format that
                       features (list[numpy.ndarray])
    labels: list of label (numpy.ndarray)


    '''

    if red_method not in REDUCTION_METHODS:
        raise ValueError('red_method {} not supported'.format(red_method))

    if config:
        input_dim = config['features']['nfilt']
    else:
        input_dim = 40  # FIXME: change this hard typed value

    # FEATURES feature reduction
    X_feat = pd.DataFrame(features).values
    labels = np.array(labels)
    is_matrix = False if np.object == X_feat.dtype else True

    if not is_matrix and (red_method in MATRIX_METHODS):
        print('all features should have the same dimension')
        sys.exit()

    if standard_scaler and is_matrix and np.isnan(X_feat).all():
        X_feat = StandardScaler().fit_transform(X_feat)

    if red_method == 'PCA' and is_matrix:
        pca = PCA(n_components=new_dimension)
        shrinked_features = pca.fit_transform(X_feat)

    elif red_method == 'LDA' and is_matrix:
        lda = my_LinearDiscriminantAnalysis(n_components=new_dimension)
        lda.fit(X_feat, labels)
        shrinked_features = lda.transform(X_feat)

    elif red_method == 'LSA' and is_matrix:
        lsa = TruncatedSVD(n_components=new_dimension, n_iter=50,
                           algorithm='arpack', random_state=42)
        shrinked_features = lsa.fit_transform(X_feat)

    elif red_method == 'TSNE' and is_matrix:
        tsne = TSNE(n_components=new_dimension, method='exact',
                    random_state=42)
        shrinked_features = tsne.fit_transform(X_feat)

    elif red_method == 'LSTM' and is_matrix:
        kr_lstm = KR_LSMTEncoder(X_feat, labels, input_dim)
        kr_lstm.fit(n_dimensions=new_dimension)
        #kr_lstm.save_data('spectral_features_variable_window.hdf5')
        shrinked_features = kr_lstm.reduce()

    elif red_method == 'AE' and is_matrix:
        kr_ae = KR_AutoEncoder(X_feat, labels)
        kr_ae.fit(n_dimensions=new_dimension)
        shrinked_features = kr_ae.reduce()

    else:  # default = raw
        shrinked_features = X_feat

    shrinked_features = [x for x in shrinked_features]

    return shrinked_features, labels


def main():
    import argparse

    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='prepare the csv or abx files to compute ABX score')

    parser.add_argument('features_source', help='csv file contining hte features')

    parser.add_argument('algorithm_config', help='algorithm configuration file for the feature extration')

    parser.add_argument('-o', '--out_csv', required=True, help='output features and labels in csv format')

    parser.add_argument('--standard_scaler', action='store_true', default=False, required=False,
            help='scale the features')

    parser.add_argument('-r', '--reduction', help=('use dimension reduction, '
                                                   'valid methods are raw, pca,lda, tsne and ae [autoencoder]'))

    args = parser.parse_args()
    data_file = args.features_source
    reduction_type = args.reduction
    config_file = args.algorithm_config
    output_csv = args.out_csv
    standard_scaler = args.standard_scaler

    # CONFIGURATION
    config = load_config(config_file)

    if reduction_type:
        red_method = reduction_type
        red_method = red_method.upper()
        if red_method not in REDUCTION_METHODS:
            print('--reduction valid algorithms: {} -- "{}" given'.format('  '.join(REDUCTION_METHODS),
                  red_method))
            sys.exit()

        try:
            new_dimension = config['dimension_reduction'][red_method]  # from the config file
        except:
            print('missing section dimension_reduction in config file [{}]'.format(red_method))
            sys.exit()

    else:
       red_method = None

    # FEATURES format:
    # label,f1,f2..fNJ where J is the total number of labels N number of features

    # FIXME: what if the format changes?
    labels_from_csv = []
    features_from_csv = []
    with open(data_file, 'r') as dfile:
        for line in dfile.readlines():
            row = line.strip().split(',')
            labels_from_csv.append(row[0])  # label/call in the first column
            features_from_csv.append([float(x) for x in row[1:]])

    # FEATURES feature reduction
    features, labels = dimension_reduction(features_from_csv, labels_from_csv,
                                           red_method, new_dimension,
                                           standard_scaler, config)

    with open(output_csv, 'w') as emb_csv:
        for label, feats in zip(labels, features):
            t = '{},'.format(label) + ','.join(['{}'.format(x)
                                                for x in feats]) + '\n'
            emb_csv.write(t)


if __name__ == '__main__':
    main()
