#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
package_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not(package_path in sys.path):
    sys.path.append(package_path)
import ABXpy.task
import ABXpy.distances.distances as distances
import ABXpy.distances.metrics.cosine as cosine
import ABXpy.distances.metrics.dtw as dtw
import ABXpy.score as score
import ABXpy.misc.items as items
import ABXpy.analyze as analyze


def cosine_distance(x, y):
    return cosine.cosine_distance(x, y)

def dtw_cosine_distance(x, y, normalized):
    return dtw.dtw(x, y, cosine.cosine_distance, normalized)


def fullrun(data_file, verbose=False, distance=cosine_distance):
    item_file = '{}.item'.format(data_file)
    feature_file = '{}.features'.format(data_file)
    distance_file = '{}.distance'.format(data_file)
    scorefilename = '{}.score'.format(data_file)
    taskfilename = '{}.abx'.format(data_file)
    analyzefilename = '{}.csv'.format(data_file)

    # deleting pre-existing files
    for f in [item_file, feature_file, distance_file,
              scorefilename, taskfilename, analyzefilename]:
        try:
            os.remove(f)
        except OSError:
            pass

    # running the evaluation:
    base=3
    n=3
    repeats=1
    n_feats=2
    max_frames=2

    items.generate_db_and_feat(base, n, repeats, item_file, n_feats, max_frames, feature_file)
    task = ABXpy.task.Task(item_file, 'call')
    task.generate_triplets(taskfilename)
    distances.compute_distances(feature_file, '/features/', taskfilename,
                                distance_file, distance,
                                normalized = True, n_cpu=1)
    score.score(taskfilename, distance_file, scorefilename)
    analyze.analyze(taskfilename, scorefilename, analyzefilename)




if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(
            prog='run_abx.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='compute ABX score')
        parser.add_argument('datafile', metavar='DATAFILE',
                            nargs=1,
                            help='file with training stimuli')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')
        return vars(parser.parse_args())

    args = parse_args()

    data_file = args['datafile'][0]
    verbose = args['verbose']
    fullrun(data_file)
