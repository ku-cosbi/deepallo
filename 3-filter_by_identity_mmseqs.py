#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
from typing import List
from tqdm import tqdm


THRESHOLD = 0.3

pdb_distance = {}
with open("../data/clean_data/pdb_distance.pkl", "rb") as f:
    pdb_distance = pickle.load(f)

Fasta = "../data/fasta_seqs/protein_sequences.fasta"
DB = "../data/fasta_seqs/protein_sequences.db"
Cluster = "../data/fasta_seqs/protein_sequences_clu"
Cluster_rep = "../data/fasta_seqs/cluster_rep"

with open(Fasta, "w") as f:
    for key, value in tqdm(pdb_distance.items()):
        f.write(f">{key}\n{value['sequence']}\n")

os.system(f"mmseqs createdb {Fasta} {DB}")
os.system(f"mmseqs cluster {DB} {Cluster} tmp --min-seq-id {THRESHOLD}")
os.system(f"mmseqs createtsv {DB} {DB} {Cluster} ../data/fasta_seqs/cluster.tsv")

os.system(f"mmseqs result2repseq {DB} {Cluster} {Cluster_rep}")
os.system(f"mmseqs result2flat {DB} {DB} {Cluster_rep} {Cluster_rep}.fasta")

cluster_rep: List[str] = []
with open(f"{Cluster_rep}.fasta", "r") as f:
    cluster_rep = f.readlines()

selected_pdbs = []
for line in cluster_rep:
    if line[0] == ">":
        line = line.strip()
        selected_pdbs.append(line[1:])

print("Number of PDBs: ", len(selected_pdbs))

pdb_diverse = {}
for pdb in selected_pdbs:
    pdb_diverse[pdb] = pdb_distance[pdb]

with open("../data/clean_data/pdb_diverse.pkl", "wb") as f:
    pickle.dump(pdb_diverse, f)

os.system("find ../data/fasta_seqs/* ! -name 'protein_sequences.fasta' ! -name 'cluster_rep.fasta' -exec rm {} +")
