#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob, os, pickle, math
from typing import List
from tqdm import tqdm
import numpy as np
from utils.pocket_feature import pocket_feature
from utils.sequence_indices import sequence_indices
from utils.pocket_coordinates import pocket_coordinates


N_ATOMS = 9
pdb_diverse = None
labels = []
features = []
seq_annotated = []
with open("../data/clean_data/pdb_diverse.pkl", "rb") as f:
    pdb_diverse = pickle.load(f)

s = len(pdb_diverse)

for pdb, info in tqdm(pdb_diverse.items()):
    modulator, mod_id, chain, residues, seq, _ = info.values()
    protein: List[str] = []
    with open(f"../data/pdbs/{pdb}.pdb", "r") as f:
        protein = f.readlines()

    # ligand center of mass
    lig_x, lig_y, lig_z, lig_cnt = 0, 0, 0, 0
    for line in protein:
        if (
            line[:6] == "HETATM" and modulator == line[17:20].strip()
            and line[21] == chain and mod_id == line[22:26].strip()
        ):
            lig_x += float(line[30:38])
            lig_y += float(line[38:46])
            lig_z += float(line[46:54])
            lig_cnt += 1

    # drop if no ligand atom found
    if lig_cnt == 0:
        continue

    lig_x /= lig_cnt
    lig_y /= lig_cnt
    lig_z /= lig_cnt

    # collect all pocket
    pocket_dir = f"../data/pockets/{pdb}_out/pockets/"
    pocket_names = glob.glob(pocket_dir + "*.pdb")
    pocket_names = sorted(
        pocket_names,
        key=lambda x: int(x.split("pocket")[-1].split("_")[0])
    )

    # for atom count
    atomTarget = {}
    for res in residues:
        atomTarget[f'{res[1]}{res[2]}'] = res[0]

    # find the nearest pocket
    dists = []

    # collect pocket features and labels
    cur_features = pocket_feature(f"../data/pockets/{pdb}_out/{pdb}_info.txt")
    selected_idxs = []
    countsPockets = [] # for atom count
    pocket_residue_indices = []

    for idx, pocket_name in enumerate(pocket_names):
        pocket: List[str] = []
        with open(pocket_name, "r") as f:
            pocket = f.readlines()
        poc_x, poc_y, poc_z, poc_cnt = 0, 0, 0, 0

        residue_indices = set()
        pocketAtomCount = 0
        for line in pocket:
            if line[:4] == "ATOM":
                poc_x += float(line[30:38])
                poc_y += float(line[38:46])
                poc_z += float(line[46:54])
                poc_cnt += 1

                # for atom count
                chainID = line[21]
                residue_index = line[22:26].strip()
                atom = line[17:20] + residue_index
                residue_indices.add(residue_index)
                if atom in atomTarget and atomTarget[atom] == chainID:
                    pocketAtomCount += 1

        # drop if no pocket atom
        if poc_cnt == 0:
            continue

        poc_x /= poc_cnt
        poc_y /= poc_cnt
        poc_z /= poc_cnt
        dist = math.sqrt(
            (poc_x - lig_x) ** 2 + (poc_y - lig_y) ** 2 +
            (poc_z - lig_z) ** 2
        )

        dists.append(dist)
        selected_idxs.append(idx)
        countsPockets.append(pocketAtomCount)
        pocket_residue_indices.append(list(residue_indices))

    # drop if there are less than 2 pockets found
    if len(dists) <= 2:
        continue

    features.append([{"pdb": pdb, "poc_emb": cur_features[idx]} for idx in selected_idxs])

    dist_min_idx = np.argmin(dists)
    # cur_labels = [0] * len(dists)
    cur_labels = [1 if item >= N_ATOMS else 0 for item in countsPockets] # for atom count
    cur_labels[dist_min_idx] = 1
    labels.append(cur_labels)

    #############################################
    seq_indices = sequence_indices(pdb, chain)
    seq_labels = ['N'] * len(seq)

    for i in range(len(cur_labels)):
        if cur_labels[i] == 1:
            for residue_index in pocket_residue_indices[i]:
                if residue_index in seq_indices and seq_indices[residue_index] < len(seq):
                    seq_labels[seq_indices[residue_index]] = 'Y'

    pocket_coord = pocket_coordinates(f"../data/pdbs/{pdb}.pdb", f"../data/pockets/{pdb}_out/pockets/", pdb, chain, pocket_residue_indices)
    seq_annotated.append({
        "pdb": pdb,
        "seq": seq,
        "chain": chain,
        "labels": ''.join(seq_labels),
        "poc_emb": [ d["poc_emb"] for d in features[-1] ],
        "poc_labels": cur_labels,
        "seq_indices": seq_indices,
        "poc_indices": pocket_residue_indices,
        "pocket_coordinates": pocket_coord
        })
    #############################################


# summarize
total_labels = sum([len(item) for item in labels])
positive_labels = sum([sum(item) for item in labels])
print(
    "total of %d pockets, with %d positive labels accounting for %.2f%%"
    % (total_labels, positive_labels, positive_labels / total_labels * 100)
)

# # clear history
os.system("rm -r ../data/classification/*")

# # # dump data
with open("../data/classification/labels_extended.pkl", "wb") as f:
    pickle.dump(labels, f)
with open("../data/classification/features_extended.pkl", "wb") as f:
    pickle.dump(features, f)
with open("../data/classification/seq_annotated.pkl", "wb") as f:
    pickle.dump(seq_annotated, f)
