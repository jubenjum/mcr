#!/usr/bin/env python

"""
extract_features: prepare the csv file with features and labels used to build ABX files.
"""

import sys

import pandas as pd
import numpy as np

from mcr.util import load_config
import mcr.load_segmented
import ipdb


def main():
    import argparse

    parser = argparse.ArgumentParser( prog=sys.argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='prepare the csv files to build abx files')

    parser.add_argument('stimuli_source', help='file with the stimuli source: wav_file, interval, label ')

    parser.add_argument('algorithm_config', help='algorithm configuration file for the feature extration')

    parser.add_argument('-o', '--out_csv', help='output features and labels in csv format')
    args = parser.parse_args()

    data_file = args.stimuli_source
    config_file = args.algorithm_config

     
    ###### CONFIGURATION
    config = load_config(config_file)
    features_params = mcr.load_segmented.ensure_list(config['features'])
    fix_stacksize = features_params['stacksize'][0]

    ###### ANNOTATIONS
    df = pd.read_csv(data_file)
    call_positions = df[['filename', 'start', 'end']].values
    labels = df['label'].values

    ###### FEATURES
    features = []

    # wav files must be sampled at 16000Hz
    encoder = mcr.load_segmented.encoder_func(features_params) # use Spectral as default
    for fname, start, end in call_positions:
        key = (fname, start)
        sig = mcr.load_segmented.load_wav(fname)
        noise = mcr.load_segmented.extract_noise(sig, features_params, encoder)

        extract_func = mcr.load_segmented.extract_features_fix_stacksize if fix_stacksize else \
                mcr.load_segmented.extract_features
        if fix_stacksize: 
            feats = extract_func(sig, noise, start, fix_stacksize, encoder)
        else:
            feats = extract_func(sig, noise, start, end, encoder)
        
        features.append(feats.flatten())


    # write the csv file
    with open(args.out_csv, 'w') as emb_csv:
        for label, feats in zip(labels, features):
            t = '{},'.format(label) + ','.join(['{}'.format(x) for x in feats]) + '\n'
            emb_csv.write(t)


if __name__ == '__main__':
    main()
