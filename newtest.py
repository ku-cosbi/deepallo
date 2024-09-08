import pickle
import torch
import random
import sys
import warnings
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor

warnings.filterwarnings('ignore')


LM_RESIDUE_PATH = "../data/classification/LM_residue_partitioned"
LM_RESIDUE_PATH = "/home/mkhokhar21/Documents/COSBI/Allostery_Paper/data/classification/LM_residue_partitioned"
THRESHOLD = 0.5
SEED = 42

def load_data(partition: str):
    pdb_data = None
    with open(f"{LM_RESIDUE_PATH}/MTL_LM_{partition}.pkl", "rb") as f:
        pdb_data = pickle.load(f)

    return pdb_data

def get_res_data(pdb_data: list):
    X = []
    Y = []

    for pdb in pdb_data:
        for i in range(min(len(pdb['poc_res_emb']), len(pdb['pocket_coordinates']))): # iterate over pockets
            seq_emb = []
            for res_idx in range(min(len(pdb['poc_res_emb'][i]), len(pdb['pocket_coordinates'][i]))):
                seq_emb.append(pdb['poc_res_emb'][i][res_idx])
            seq_emb = np.array(seq_emb).mean(axis=0)
            poc = pdb["poc_emb"][i]
            X.append(np.concatenate((seq_emb, poc)))
            # X.append(seq_emb)
            Y.append(pdb["poc_labels"][i])

    return X, Y

pdb_data = load_data("train")
pdb_data.extend(load_data("test"))

pdb_list = ['2FPL', '2R1R', '3BCR', '4PFK', '1Q5O', '3PEE', '4HO6', '1XMV', '2OZ6', '3LNY', '5DKK']
found_pdbs = []

for x in pdb_data:
    if x['pdb'] in pdb_list:
        found_pdbs.append(x['pdb'])

# Get missing PDBs
missing_pdbs = [pdb for pdb in pdb_list if pdb not in found_pdbs]

print("Missing PDBs:", missing_pdbs)
