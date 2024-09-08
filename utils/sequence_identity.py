#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from Bio import Align


def sequence_identity(seq1, seq2):
    # ref: https://github.com/cyrusmaher/mosaic/blob/master/mosaic.py#L383

    aligner = Align.PairwiseAligner()
    alignments = aligner.align(seq1, seq2)
    seq1_aligned = alignments[0][0]
    seq2_aligned = alignments[0][1]

    seq1_aligned = np.array([char for char in seq1_aligned])
    seq2_aligned = np.array([char for char in seq2_aligned])

    seq1_notmissing = np.logical_and(
        (seq1_aligned != 'U'), (seq1_aligned != 'X')
    )
    seq2_notmissing = np.logical_and(
        (seq2_aligned != 'U'), (seq2_aligned != 'X')
    )

    pair_matches = (seq1 == seq2)

    quality_matches = np.logical_and(pair_matches, seq1_notmissing)
    quality_comparisons = np.logical_and(seq1_notmissing, seq2_notmissing)
    return 1.0 * quality_matches.sum() / (quality_comparisons.sum())
