#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import torch
import requests
import glob, os, math
import numpy as np
from transformers import BertModel, BertTokenizer
import xgboost as xgb

from utils.extract_sequence import extract_sequence
from utils.pocket_feature import pocket_feature
from utils.sequence_indices import sequence_indices
from utils.pocket_coordinates import pocket_coordinates

N_ATOMS = 9
MODEL_PATH = "../prot_bert_allosteric"
base_url = "https://files.rcsb.org/download"
pdb_dir = "../data/pdbs/"
pocket_dir = "../data/pockets/"
pdb_id = "3BCR"
chain_id = "A"

TEST = False

pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
pocket_path = os.path.join(pocket_dir, f"{pdb_id}_out")

#### Test - begin ####
if TEST:
    ASD_path = "../data/source_data/ASD_Release_201909_AS.txt"

    asd = None
    with open(ASD_path, "r") as f:
        asd = f.readlines()

    mod_id, modulator, residues = None, None, None
    for line in asd[1:]:
        line = line.strip().split("\t")
        pdb, modulator, chain_id, mod_id = line[4], line[6], line[7], line[11]

        if pdb != pdb_id:
            continue

        if len(set(chain_id.split(";"))) != 1:
            continue
        chain_id = chain_id[0]

        if len(set(modulator.split(";"))) != 1:
            continue
        modulator = modulator.split(";")[0]

        # extract residues
        res_raw = [
            res.replace(":", ",").split(",") for res in line[-1].split("; ")
        ]
        # residue_clean format: chain id + residue type + residue number
        residues = [
            [res[0][-1], ch[:3], ch[3:]] for res in res_raw for ch in res[1:]
        ]
        # select only residues in the same chain of modulator
        residues = [res for res in residues if res[0] == chain_id]

        break
#### Test - end ####


if not os.path.exists(pdb_path):
    response = requests.get(f"{base_url}/{pdb_id}.pdb")
    if response.status_code == 200:  # Check if the request was successful
        with open(pdb_path, 'wb') as file:
            file.write(response.content)
        print(f"PDB file {pdb_id}.pdb downloaded successfully.")
    else:
        raise Exception(f"Failed to download {pdb_id}.pdb. Check if the PDB ID is correct.")

sequence = extract_sequence(pdb_path, chain_id)

if len(sequence) <= 10:
    raise Exception("Sequence is too short.")

if not os.path.exists(pocket_path):
    os.system(f"fpocket -f {pdb_path} -k {chain_id}")
    os.system(f"mv {os.path.join(pdb_dir, pdb_id)}_out {pocket_dir}")

#### Test - begin ####
if TEST:
    protein = None
    lig_x, lig_y, lig_z, lig_cnt = 0, 0, 0, 0

    with open(pdb_path, "r") as f:
        protein = f.readlines()

    for line in protein:
        if (
            line[:6] == "HETATM" and modulator == line[17:20].strip()
            and line[21] == chain_id and mod_id == line[22:26].strip()
        ):
            lig_x += float(line[30:38])
            lig_y += float(line[38:46])
            lig_z += float(line[46:54])
            lig_cnt += 1

    lig_x /= lig_cnt
    lig_y /= lig_cnt
    lig_z /= lig_cnt
#### Test - end ####

pocket_names = glob.glob(f"{pocket_path}/pockets/*.pdb")
pocket_names = sorted(
    pocket_names,
    key=lambda x: int(x.split("pocket")[-1].split("_")[0])
)

pockets_feats = pocket_feature(f"{pocket_path}/{pdb_id}_info.txt")
selected_idxs = []
pocket_residue_indices = []

#### Test - begin ####
if TEST:
    atomTarget = {}
    for res in residues:
        atomTarget[f'{res[1]}{res[2]}'] = res[0]

    dists = []
    countsPockets = [] # for atom count
#### Test - end ####

for idx, pocket_name in enumerate(pocket_names):
    pocket = None
    with open(pocket_name, "r") as f:
        pocket = f.readlines()

#### Test - begin ####
    if TEST:
        poc_x, poc_y, poc_z = 0, 0, 0
        pocketAtomCount = 0
#### Test - end ####

    poc_cnt = 0
    residue_indices = set()

    for line in pocket:
        if line[:4] == "ATOM":
            poc_cnt += 1
            residue_index = line[22:26].strip()
            atom = line[17:20] + residue_index
            residue_indices.add(residue_index)

#### Test - begin ####
            if TEST:
                poc_x += float(line[30:38])
                poc_y += float(line[38:46])
                poc_z += float(line[46:54])
                chainID = line[21]
                if atom in atomTarget and atomTarget[atom] == chainID:
                    pocketAtomCount += 1
#### Test - end ####

    if poc_cnt == 0:
        continue

#### Test - begin ####
    if TEST:
        poc_x /= poc_cnt
        poc_y /= poc_cnt
        poc_z /= poc_cnt
        dist = math.sqrt(
            (poc_x - lig_x) ** 2 + (poc_y - lig_y) ** 2 +
            (poc_z - lig_z) ** 2
        )

        dists.append(dist)
        countsPockets.append(pocketAtomCount)
#### Test - end ####

    selected_idxs.append(idx)
    pocket_residue_indices.append(list(residue_indices))

if len(selected_idxs) <= 2:
    raise Exception("Too few pockets extracted.")

pocket_features = [pockets_feats[idx] for idx in selected_idxs]

seq_indices = sequence_indices(pdb_id, chain_id)

#### Test - begin ####
if TEST:
    dist_min_idx = np.argmin(dists)
    labels = [1 if item >= N_ATOMS else 0 for item in countsPockets] # for atom count
    labels[dist_min_idx] = 1

    seq_labels = ['N'] * len(sequence)
    for i in range(len(labels)):
            if labels[i] == 1:
                for residue_index in pocket_residue_indices[i]:
                    if residue_index in seq_indices and seq_indices[residue_index] < len(sequence):
                        seq_labels[seq_indices[residue_index]] = 'Y'
#### Test - end ####

pocket_coord = pocket_coordinates(pdb_path, f"{pocket_path}/pockets/", pdb_id, chain_id, pocket_residue_indices)

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, do_lower_case=False )
model = BertModel.from_pretrained(MODEL_PATH)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

seq_emb = None
poc_res_emb = []

#### Test - begin ####
if TEST:
    poc_labels = []
#### Test - end ####

with torch.no_grad():
    seq = " ".join(sequence)
    encoding = tokenizer.batch_encode_plus(
        [seq],
        add_special_tokens=True,
        padding='max_length'
    )
    input_ids = torch.tensor(encoding['input_ids']).to(device)
    attention_mask = torch.tensor(encoding['attention_mask']).to(device)
    embedding = model(input_ids=input_ids, attention_mask=attention_mask)[0].cpu().numpy()

    seq_len = (attention_mask[0] == 1).sum()
    token_emb = embedding[0][1:seq_len-1]
    seq_emb = token_emb.mean(axis=0)

    for i in range(len(pocket_residue_indices)):
        add_pocket = True
        cur_poc_emb = []

#### Test - begin ####
        if TEST:
            poc_labels.append(labels[i])
#### Test - end ####

        for idx in pocket_residue_indices[i]:
            try:
                token = token_emb[seq_indices[idx]]
                cur_poc_emb.append(token)
            except Exception as e:
                add_pocket = False
#### Test - begin ####
                if TEST:
                    poc_labels.pop()
#### Test - end ####
                break
        
        if add_pocket:
            poc_res_emb.append(cur_poc_emb)

del model
torch.cuda.empty_cache()
gc.collect()

def get_res_data(poc_res_emb, pocket_coord):
    X = []
    Y = []

    for i in range(min(len(poc_res_emb), len(pocket_coord))):
        seq_emb = []
        for res_idx in range(min(len(poc_res_emb[i]), len(pocket_coord[i]))):
            seq_emb.append(poc_res_emb[i][res_idx])
        seq_emb = np.array(seq_emb).mean(axis=0)
        poc = pocket_features[i]
        X.append(np.concatenate((seq_emb, poc)))
#### Test - begin ####
        if TEST:
            Y.append(labels[i])
#### Test - end ####

    return X, Y

X_Test, Y_Test = get_res_data(poc_res_emb, pocket_coord)
dtest = xgb.DMatrix(X_Test, label=Y_Test)
bst = xgb.Booster()
bst.load_model('./xgboost.model')
y_pred = bst.predict(dtest)

paired = list(zip(y_pred, Y_Test, pocket_residue_indices))
paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)

top3 = [paired_sorted[i] for i in range(min(len(paired_sorted), 3))]

for top in top3:
    print(top)
