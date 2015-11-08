"""eval_continuous: evaluate continuous call recognition, i.e. transcription

measures:
1. UTTERANCE - word labeling
2. FRAME - frame labeling
3. CALL - call location .

ad 1.
let H = the number of correct labels for calls
let N = the number of labels in the gold file
let I = the number of insertions
let D = the number of deletions
then WER = (I+D+S)/N
     WAC = 1 - WER = (N-S-D-I)/N = (H-I)/N
     MER = (S+D+I)/(H+S+D+I) = 1 - MAC
     MAC = H/(H+S+D+I)
also report H, I, D and N

ad 2.
define precision, recall and f1-score from frame labeling

ad 3.

count as TP a predicted call with boundaries within tolerance of gold call
count as FP a predicted call that is not within tolerance of a gold call
count as FN a gold call with no predicted call within tolerance

then CALL_LOCATION_PRECISION = TP/(TP+FP)
and  CALL_LOCATION_RECALL    = TP/(TP+FN)

"""

from __future__ import division

import numpy as np
import pandas as pd
from collections import namedtuple
from itertools import dropwhile, takewhile
from sklearn.metrics import classification_report

from mcr.util import verb_print


class Span(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def intersect(self, other):
        return Span(max(self.start, other.start),
                    min(self.end, other.end))

    def __len__(self):
        return self.end - self.start

class Interval(object):
    def __init__(self, label, span):
        self.label = label
        self.span = span


def find_nearest_ix(arr, query):
    ix = np.searchsorted(arr, query, side='left')
    if ix == 0:
        found_ix = 0
    elif ix == arr.shape[0]:
        found_ix = -1
    elif query - arr[ix-1] < arr[ix] - query:
        found_ix = ix - 1
    else:
        found_ix = ix
    return found_ix


def call_score(gold, pred, tolerance):
    TP = FP = FN = 0
    for filename in gold.filename.unique():
        pred_starts = pred[pred.filename == filename].start.values
        pred_ends = pred[pred.filename == filename].end.values
        pred_labels = pred[pred.filename == filename].label.values
        for _, row in gold[(gold.label != 'SIL') &
                           (gold.filename == filename)].iterrows():
            query_start = row.start
            query_end = row.end
            query_label = row.label
            # best match could be from front or back
            # FRONT:
            match_ix = find_nearest_ix(pred_starts, query_start)
            match_start = pred_starts[match_ix]
            match_end = pred_ends[match_ix]
            match_label = pred_labels[match_ix]

            if abs(match_start - query_start) < tolerance and \
               abs(match_end - query_end) < tolerance and \
               match_label == query_label:
                TP += 1
                continue

            # BACK:
            match_ix = find_nearest_ix(pred_ends, query_end)
            match_start = pred_starts[match_ix]
            match_end = pred_ends[match_ix]
            match_label = pred_labels[match_ix]

            if abs(match_start - query_start) < tolerance and \
               abs(match_end - query_end) < tolerance and \
               match_label == query_label:
                TP += 1
                continue

            FN += 1
    for filename in pred.filename.unique():
        gold_starts = gold[gold.filename == filename].start.values
        gold_ends = gold[gold.filename == filename].end.values
        gold_labels = gold[gold.filename == filename].label.values
        for _, row in pred[(pred.label != 'SIL') &
                           (pred.filename == filename)].iterrows():
            query_start = row.start
            query_end = row.end
            query_label = row.label
            # best match could be from front or back
            # FRONT:
            match_ix = find_nearest_ix(gold_starts, query_start)
            match_start = gold_starts[match_ix]
            match_end = gold_ends[match_ix]
            match_label = gold_labels[match_ix]

            if abs(match_start - query_start) < tolerance and \
               abs(match_end - query_end) < tolerance and \
               match_label == query_label:
                continue

            # BACK:
            match_ix = find_nearest_ix(gold_ends, query_end)
            match_start = gold_starts[match_ix]
            match_end = gold_ends[match_ix]
            match_label = gold_labels[match_ix]

            if abs(match_start - query_start) < tolerance and \
               abs(match_end - query_end) < tolerance and \
               match_label == query_label:
                continue

            FP += 1

    return TP/(TP+FP), TP/(TP+FN)


def utterance_ops(XA, XB):
    """Return the number of opcodes needed for transforming sequence XA into XB.

    Parameters
    ----------
    XA, XB : sequence

    Returns
    -------
    dict with keys:
      H: correct
      I: insertions
      D: deletions
      S: substitutions
      N: length of A
    """
    mA, mB = len(XA), len(XB)
    M = np.zeros((mA+1, mB+1))
    M[:, 0] = np.arange(0, mA+1)
    M[0, :] = np.arange(0, mB+1)
    B = np.zeros((mA+1, mB+1))
    B[:, 0] = 3
    B[0, :] = 4
    # backpointers:
    # 1 = diagonal - equal
    # 2 = diagonal - substitution
    # 3 = vertical - insertion
    # 4 = horizontal - deletion

    for i in xrange(1, mA+1):
        for j in xrange(1, mB+1):
            if XA[i-1] == XB[j-1]:
                M[i, j] = M[i-1, j-1]
                B[i, j] = 1
            else:
                costs = [M[i-1, j-1], M[i-1, j], M[i, j-1]]
                argmin = np.argmin(costs)
                B[i, j] = argmin + 2
                M[i, j] = costs[argmin] + 1

    i = mA; j = mB
    H = I = D = S = 0
    while i > 0 or j > 0:
        move = B[i, j]
        if move == 0:
            break
        elif move == 1:
            H += 1
            i -= 1
            j -= 1
        elif move == 2:
            S += 1
            i -= 1
            j -= 1
        elif move == 3:
            D += 1
            i -= 1
        elif move == 4:
            I += 1
            j -= 1
    return OpcodeCounter(
        H=H,
        I=I,
        D=D,
        S=S,
        N=mA
    )


class OpcodeCounter(dict):
    _keys = {'H', 'I', 'D', 'S', 'N'}

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            if key in self._keys:
                return 0
            else:
                raise AttributeError(key)

    def __getstate__(self):
        return self.__dict__

    def __add__(self, other):
        if not isinstance(other, OpcodeCounter):
            raise ValueError('cannot add OpcodeCounter and {}'.format(
                other.__class__))
        return OpcodeCounter(
            H=self.H + other.H,
            I=self.I + other.I,
            D=self.D + other.D,
            S=self.S + other.S,
            N=self.N + other.N
        )

    def __iadd__(self, other):
        if not isinstance(other, OpcodeCounter):
            raise ValueError('cannot add OpcodeCounter and {}'.format(
                other.__class__))
        self.H += other.H
        self.I += other.I
        self.D += other.D
        self.S += other.S
        self.N += other.N
        return self


def utterance_error_rates(ops):
    H, D, I, S, N = ops.H, ops.D, ops.I, ops.S, ops.N
    WER = (S+D+I)/N
    MAC = H/(H+S+D+I)
    MER = 1 - MAC
    MER = (S+D+I)/(H+S+D+I)
    return WER, MER


def utterance_score(gold, pred):
    # split out by file
    gold_utts = {
        name: list(group.label.values)
        for name, group in gold[gold.label != 'SIL'].groupby(gold.filename)
    }
    pred_utts = {
        name: list(group.label.values)
        for name, group in pred[pred.label != 'SIL'].groupby(pred.filename)
    }

    gold_keys = set(gold_utts.keys())
    pred_keys = set(pred_utts.keys())

    # check that there are not more predicted than gold filenames
    assert len(pred_keys - gold_keys) == 0

    counter = OpcodeCounter()
    for filename in gold_keys:
        gold_utt = gold_utts[filename]
        if filename in pred_keys:
            pred_utt = pred_utts[filename]
        else:
            pred_utt = []
        counter += utterance_ops(gold_utt, pred_utt)
    WER, MER = utterance_error_rates(counter)
    counter['WER'] = WER
    counter['MER'] = MER
    return counter


def frame_score(gold, pred, winshift):
    gold_labels = set(gold.label.unique())
    pred_labels = set(pred.label.unique())
    assert (len(pred_labels) - len(gold_labels) == 0)

    label2ix = {
        label: ix
        for ix, label in enumerate(sorted(gold_labels))
    }

    y_gold_all = None
    y_pred_all = None

    for filename in gold.filename:
        gold_file = gold[gold.filename == filename]
        pred_file = pred[pred.filename == filename]
        assert len(pred_file) > 0
        # make sure there are no gaps in either gold or predicted ann.
        assert all(np.isclose(gold_file.iloc[i].end, gold_file.iloc[i+1].start)
                    for i in xrange(len(gold_file)-1))
        assert all(np.isclose(pred_file.iloc[i].end, pred_file.iloc[i+1].start)
                   for i in xrange(len(pred_file)-1))

        gold_ivals = [Interval(row.label, Span(row.start, row.end))
                      for _, row in gold_file.iterrows()]
        pred_ivals = [Interval(row.label, Span(row.start, row.end))
                      for _, row in pred_file.iterrows()]
        gold_span = Span(gold_ivals[0].span.start, gold_ivals[-1].span.end)
        pred_span = Span(pred_ivals[0].span.start, pred_ivals[-1].span.end)
        span = gold_span.intersect(pred_span)

        # truncate lists
        strip = lambda x: list(
            takewhile(lambda ival: ival.span.start < span.end,
                      dropwhile(lambda ival: ival.span.end < span.start, x)
            )
        )
        gold_span = strip(gold_ivals)
        pred_span = strip(pred_ivals)

        gold_ivals[0].span.start = span.start
        gold_ivals[-1].span.end = span.end
        pred_ivals[0].span.start = span.start
        pred_ivals[-1].span.end = span.end

        n_values = int(len(span) / winshift)
        y_gold = np.zeros(n_values, dtype=np.uint8) - 1
        y_pred = np.zeros(n_values, dtype=np.uint8) - 1

        gold_ptr = 0
        pred_ptr = 0
        for val_ptr in xrange(n_values):
            t = val_ptr * winshift
            if t > gold_ivals[gold_ptr].span.end:
                gold_ptr += 1
                if gold_ptr >= len(gold_ivals):
                    raise ValueError('gold_ptr advanced too far')
            if t > pred_ivals[pred_ptr].span.end:
                pred_ptr += 1
            gold_label = gold_ivals[gold_ptr].label
            pred_label = pred_ivals[pred_ptr].label
            y_gold[val_ptr] = label2ix[gold_label]
            y_pred[val_ptr] = label2ix[pred_label]
        if y_gold_all is None:
            y_gold_all = y_gold
            y_pred_all = y_pred
        else:
            y_gold_all = np.hstack((y_gold_all, y_gold))
            y_pred_all = np.hstack((y_pred_all, y_pred))

    return classification_report(
        y_gold_all, y_pred_all,
        target_names=sorted(gold_labels)
    )


def quantize_calls(df, winshift):
    # convert to frames
    df['start_q'] = np.round(df['start'].values / winshift)*winshift
    df['end_q'] = np.round(df['end'].values / winshift)*winshift


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(
            prog='eval_continuous.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='evaluate continuous recognition'
        )
        parser.add_argument(
            'goldfile',
            metavar='GOLDFILE',
            nargs=1,
            help='file with gold stimuli'
        )
        parser.add_argument(
            'predfile',
            metavar='PREDFILE',
            nargs=1,
            help='file with predicted stimuli'
        )
        parser.add_argument(
            '-t', '--tolerance',
            action='store',
            dest='tolerance',
            default=0.5,
            help='boundary tolerance in seconds'
        )
        parser.add_argument(
            '-s', '--winshift',
            action='store',
            dest='winshift',
            default=0.01,
            help='frame shift in seconds'
        )
        parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            dest='verbose',
            default=False,
            help='talk more'
        )

        return vars(parser.parse_args())

    args = parse_args()

    gold_file = args['goldfile'][0]
    pred_file = args['predfile'][0]
    tolerance = float(args['tolerance'])
    winshift = float(args['winshift'])
    verbose = args['verbose']

    with verb_print('reading gold stimuli from {}'.format(gold_file), verbose):
        gold_df = pd.read_csv(gold_file)
        quantize_calls(gold_df, winshift)
    with verb_print('reading predicted stimuli from {}'
                    .format(pred_file), verbose):
        pred_df = pd.read_csv(pred_file)
        quantize_calls(pred_df, winshift)

    with verb_print('calculating utterance score', verbose):
        oc = utterance_score(gold_df, pred_df)

    with verb_print('calculating call score (tolerance={:.3f})'
                    .format(tolerance), verbose):
        prec, rec = call_score(gold_df, pred_df, tolerance)

    with verb_print('calculating frame score (window shift={:.3f})'
                    .format(winshift), verbose):
        frame_report = frame_score(gold_df, pred_df, 0.01)

    print
    print '='*53
    print 'UTTERANCE:'
    print '   WER: {wer:.2f}%, MER: {mer:.2f}%'.format(
        wer=oc.WER*100,
        mer=oc.MER*100
    )
    print '   [H={H},D={D},S={S},I={I},N={N}]'.format(
        H=oc.H,
        D=oc.D,
        S=oc.S,
        I=oc.I,
        N=oc.N
    )

    print '='*53
    print 'CALL:'
    print '   precision: {0:.3f}, recall: {1:.3f}, f-score: {2:.3f}'.format(
        prec, rec, 2*prec*rec/(prec+rec)
    )

    print '='*53
    print 'FRAME:'
    print frame_report
    print '='*53