#!/usr/bin/env python

"""
extract_features: prepare the csv file with features and labels used to build ABX files.
"""

import sys
from collections import Counter


import pandas as pd
import numpy as np

from mcr.util import load_config
import mcr.load_segmented
from mcr.util import build_cache
from mcr.util import normalize 

memory = build_cache()

__all__ = ['get_features', 'save_features']


def save_features(fname, features, labels, sep=','):
    ''' save_features save the features and labels in a csv file using
    pandas to_csv function.

    First column will be stored the labels, and the rest of columns
    contains the features.

    Parameters
    ----------
    
    fname: [string]
        output file name

    features : [list or np.array]
        The list of features that will be saved, it can be a python list 
        or np.array, the elements must be floating point values

    labels : [list or np.array]
        labels is a python or numpy list with any printable object 

    sep : [string, default=',']
        csv field separation 


    Returns
    -------
    
    A csv file with the name set by fname variable

    '''
    new_labels = np.array([labels]).T
    _, num_labels = new_labels.shape
    new_features = np.array(features, dtype=np.float64)
    _, num_features = new_features.shape
    df = pd.DataFrame(np.hstack((new_labels, new_features)))
    df.to_csv(fname, sep=sep)


#@memory.cache
def get_features(features_params, call_intervals, read_labels):
    ''' get_features reads the wav files and returns the features
    for the selected intervals from the annotation file

    Parameters
    ----------
    
    features_params : [dict]
         parameters from the config file, loaded with mcr.util.load_config
    
    call_intervals : [np.ndarray]
         arrays with structure = [wav_filename, start_time, end_time] 

    read_labels : [list of Strings]
         labels for the call_intervals, a python list with string elements

    Returns
    -------
    
    features : [list of np.ndarrays] 
        features extracted from call_intervals files using parameters from
        the features_params, a list of numpy arrays is returned as it can
        fit different size features 
              
    labels : [np.darray] 
        Features' labels  

    '''

    # if fix-stacksize is set to 0 it it will read all the interval from the transcriptions
    fix_stacksize = features_params['stacksize'][0]

    # FEATURES
    features_ = []

    # sampling rate can be variable, but mainly I used 16000Hz 
    encoder = mcr.load_segmented.encoder_func(features_params) # use Spectral as default
    for fname, start, end in call_intervals:
        key = (fname, start)
        sig = mcr.load_segmented.load_wav(fname, encoder.fs)

        noise = mcr.load_segmented.extract_noise(sig, features_params, encoder)

        # select the the specific function to extract features: 
        # one that deals only with fix size stack (develped by Maarten) and
        # other that extracts features when the the stack size is not constant
        extract_func = mcr.load_segmented.extract_features_fix_stacksize \
                       if fix_stacksize else mcr.load_segmented.extract_features

        if fix_stacksize:
            feats = extract_func(sig, noise, start, fix_stacksize, encoder)
        else:
            try:
                feats = extract_func(sig, noise, start, end, encoder)
            except:
                pass

        features_.append(feats.flatten())

    return features_, read_labels


def main():
    import argparse

    help_switch = """Allow to replace the default features extracted with 
    parameters on `config_file` for a hard-coded features.
    The format of the file is csv with fields separated by ","
    and file should contain: filename,start,end,label,new_features
    NOTE: csv without header
    """

    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='prepare the csv files to build abx files')

    parser.add_argument('annotations_file', help='file with the stimuli source: wav_file, interval, label ')

    parser.add_argument('config_file', help='algorithm configuration file for the feature extration')
    
    parser.add_argument('-s', '--switch_features', required=False,                   
                        help=help_switch)

    parser.add_argument('-o', '--out_csv', default='features',
            help='output features and labels in csv format')

    parser.add_argument('-n', '--normalize', action='store_true', default=False, required=False, 
                        help='normalize and fill with zeros the features that are variable size')

    args = parser.parse_args()

    annotation_file = args.annotations_file
    config_file = args.config_file
    switch_file = args.switch_features
    normalization = args.normalize

    # CONFIGURATION
    config = load_config(config_file)
    features_params = mcr.load_segmented.ensure_list(config['features'])

    # READ ANNOTATIONS & GET FEATURES
    df = pd.read_csv(annotation_file)
    call_intervals = df[['filename', 'start', 'end']].values
    read_labels = df['label'].values
    features, labels = get_features(features_params, call_intervals,
                                    read_labels)

    if normalization:
        features = normalize(pd.DataFrame(features).values)

    # read the new features, I am reading line by line because the number of 
    # features can be different from row to row
    new_features = {}
    if switch_file:
        with open(switch_file, 'r') as sfile:
            for line in sfile.readlines():
                line = line.strip().split(',')
                newfile = line[0]
                start = float(line[1])
                end = float(line[2])
                label = line[3]
                try:
                     idx = df.query(('(filename == @newfile) and ' 
                         '(label == @label) and' 
                         '(@end >= start and @start <= end)')).index[0]
                except IndexError:
                    print('new feature not found: {} {} {} {}'.format(newfile, start, end, label))
                    sys.exit()
                new_features[idx] = line[4:]


    # write the csv file
    with open(args.out_csv, 'w') as emb_csv:
        for idx, (label, feats) in enumerate(zip(labels, features)):
            try:
                feats_ = new_features[idx]
            except KeyError:
                feats_ = feats
                
            t = '{},'.format(label) + ','.join(['{}'.format(x) for x in feats_]) + '\n'
            emb_csv.write(t)


if __name__ == '__main__':
    main()
