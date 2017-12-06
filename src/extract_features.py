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

    ###### ANNOTATIONS
    df = pd.read_csv(data_file)
    X = df[['filename', 'start', 'end']].values
    labels = df['label'].values


    ###### FEATURES
    feat_cache = {}
    # wav files must be sampled at 16000Hz
    encoder = mcr.load_segmented.encoder_func(features_params) # use Spectral as default
    for fname, start, end in X:
        key = (fname, start)
        sig = mcr.load_segmented.load_wav(fname)
        noise = mcr.load_segmented.extract_noise(sig, features_params, encoder)
        if features_params['stacksize'][0] != 0:
            end = start + (features_params['stacksize'][0] * features_params['window_shift'][0] ) + \
                    features_params['window_length'][0] / 2.0

        feat_cache[key] = mcr.load_segmented.extract_features(sig, noise, start, end, \
                encoder, buffer_length=0.1)

    ### transforming the embeddings 
    all_features = list()
    all_file_descriptions = list()
    all_calls = list()
    for file_description, features_selection in feat_cache.iteritems():
        # the calls in the same order that the embeddings
	filename, start_time = file_description
        all_calls.append(df.label.loc[(df['filename'] == filename) &
                 (df['start'] == start_time)].values[0])

        # building the embedding 
        v_ = np.array(features_selection.flatten())
        all_features.append(v_)
        all_file_descriptions.append(file_description)

    small_embeddings = np.array(all_features)

    # write the csv file
    with open(args.out_csv, 'w') as emb_csv:
        for n, file_description in enumerate(all_file_descriptions):
            try:
                sel = small_embeddings[n,:] # when it is a numpy array
            except:
                sel = small_embeddings[n]

            t = '"{}",'.format(all_calls[n]) + \
                 ','.join(['{}'.format(x) for x in sel]) + '\n'
            emb_csv.write(t)


if __name__ == '__main__':
    main()
