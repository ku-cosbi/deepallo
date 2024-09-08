#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from tqdm import tqdm
from utils.sequence_identity import sequence_identity


THRESHOLD = 0.25

pdb_distance = None
with open("../data/clean_data/pdb_distance.pkl", "rb") as f:
    pdb_distance = pickle.load(f)
pdb_seq_dist = []
for pdb in pdb_distance.keys():
    pdb_seq_dist.append(
        [pdb, pdb_distance[pdb]["distance"], pdb_distance[pdb]["sequence"]]
    )

pdb_seq_dist.sort(key=lambda x: x[1])

sequences = [pdb_seq_dist[0][2]]
selected_pdbs = [pdb_seq_dist[0][0]]

for pdb, _, seq in tqdm(pdb_seq_dist[1:]):
    identity_percent = list(
        map(sequence_identity, [seq] * len(sequences), sequences)
    )
    if max(identity_percent) <= THRESHOLD:
        sequences.append(seq)
        selected_pdbs.append(pdb)


print("Number of PDBs: ", len(selected_pdbs))

pdb_diverse = {}
for pdb in selected_pdbs:
    pdb_diverse[pdb] = pdb_distance[pdb]

with open("../data/clean_data/pdb_diverse.pkl", "wb") as f:
    pickle.dump(pdb_diverse, f)
