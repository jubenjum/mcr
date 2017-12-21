#!/usr/bin/env python

"""
extract_features: prepare the csv file with features and labels used to build ABX files.
"""

import sys

import pandas as pd
import numpy as np

from mcr.util import load_config
import mcr.load_segmented


__all__ = ['get_features', 'save_features']


def save_features(fname, features, labels, sep=','):
    '''
    Parameters
    ----------
    
    fname : file, str
        file name

    features : list
        features/embeddings that will be save

    labels : list
        labels for the features

    sep : str [default=',']
        field separation


    Returns
    -------
    None
    
    '''
    new_labels = np.array([labels]).T
    _, num_labels = new_labels.shape
    new_features = np.array(features, dtype=np.float64)
    _, num_features = new_features.shape
    df = pd.DataFrame(np.hstack((new_labels, new_features)))
    df.to_csv(fname, sep=sep)


def get_features(features_params, call_intervals, read_labels):
    ''' get_features reads the wav files and returns the features
    for the selected intervals from the annotation file

    Parameters
    ----------
    features_params : parameters from the config file, loaded with mcr.util.load_config (dict)
    call_intervals : arrays with structure = [wav_filename, start_time, end_time  ] (numpy.ndarray)
    read_labels : labels for the call_intervals (numpy.ndarray)

    Returns
    -------
    features: features extracted from call_intervals files using parameters from 
              the features_params, a list of numpy arrays is returned as it can
              fit different size features (list[numpy.ndarray])
    labels: labels related to the features (numpy.ndarray)

    '''

    ### if fix-stacksize is set to 0 it it will read all the interval from the transcriptions
    fix_stacksize = features_params['stacksize'][0]

    ###### FEATURES
    features_ = []

    # in the pipeline wav files are at 16000Hz mono
    encoder = mcr.load_segmented.encoder_func(features_params) # use Spectral as default
    for fname, start, end in call_intervals:
        key = (fname, start)
        sig = mcr.load_segmented.load_wav(fname)
        noise = mcr.load_segmented.extract_noise(sig, features_params, encoder)

        extract_func = mcr.load_segmented.extract_features_fix_stacksize if fix_stacksize else \
                mcr.load_segmented.extract_features
        if fix_stacksize: 
            feats = extract_func(sig, noise, start, fix_stacksize, encoder)
        else:
            feats = extract_func(sig, noise, start, end, encoder)
        
        features_.append(feats.flatten())
    
    return features_, read_labels


def main():
    import argparse

    parser = argparse.ArgumentParser( prog=sys.argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='prepare the csv files to build abx files')

    parser.add_argument('annotations_file', help='file with the stimuli source: wav_file, interval, label ')

    parser.add_argument('config_file', help='algorithm configuration file for the feature extration')

    parser.add_argument('-o', '--out_csv', help='output features and labels in csv format')
    
    args = parser.parse_args()

    annotation_file = args.annotations_file
    config_file = args.config_file

    ###### CONFIGURATION
    config = load_config(config_file)
    features_params = mcr.load_segmented.ensure_list(config['features'])
    
    ###### READ ANNOTATIONS & GET FEATURES
    df = pd.read_csv(annotation_file)
    call_intervals = df[['filename', 'start', 'end']].values
    read_labels = df['label'].values
    features, labels = get_features(features_params, call_intervals, read_labels)

    # write the csv file
    with open(args.out_csv, 'w') as emb_csv:
        for label, feats in zip(labels, features):
            t = '{},'.format(label) + ','.join(['{}'.format(x) for x in feats]) + '\n'
            emb_csv.write(t)


if __name__ == '__main__':
    main()
