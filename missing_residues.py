#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from tqdm import tqdm
from utils.sequence_identity import sequence_identity
from Bio.PDB import *

THRESHOLD = 0.25

labels = pickle.load(open("../data/classification/labels_extended.pkl", "rb"))

# summarize
total_labels = sum([len(item) for item in labels])
positive_labels = sum([sum(item) for item in labels])
print(
    "total of %d pockets, with %d positive labels accounting for %.2f%%"
    % (total_labels, positive_labels, positive_labels / total_labels * 100)
)