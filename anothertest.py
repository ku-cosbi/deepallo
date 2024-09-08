import pickle
import torch
import random
import sys
import warnings

LM_RESIDUE_PATH = "/home/mkhokhar21/Documents/COSBI/Allostery_Paper/data/classification/LM_residue_partitioned"

def load_data(partition: str):
    pdb_data = None
    with open(f"{LM_RESIDUE_PATH}/MTL_LM_{partition}.pkl", "rb") as f:
        pdb_data = pickle.load(f)

    return pdb_data

pdb_data = load_data("train")
train_len = 0
for pdb in pdb_data:
    train_len += len(pdb['seq'])

pdb_data = load_data("test")
test_len = 0
for pdb in pdb_data:
    test_len += len(pdb['seq'])

s = ''
