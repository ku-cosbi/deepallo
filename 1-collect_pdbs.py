#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import requests, pickle, os
from tqdm import tqdm
from utils.extract_sequence import extract_sequence


# pdb urls
rcsb_download = "https://files.rcsb.org/download/"
rcsb_structure = "https://www.rcsb.org/structure/"

pdb_info = {}

# ASD data
asd_dir = "../data/source_data/ASD_Release_201909_AS.txt"
asd: List[str] = []
with open(asd_dir, "r") as f:
    asd = f.readlines()

for line in tqdm(asd[1:]):
    line = line.strip().split("\t")
    pdb, modulator, chain, mod_id = line[4], line[6], line[7], line[11]

    # drop PDBs that have modulators in multiple distinct chains
    if len(set(chain.split(";"))) != 1:
        continue
    chain = chain[0]

    # drop PDBs that have different modulators
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
    residues = [res for res in residues if res[0] == chain]

    # drop the current pdb if it's seen and does not have more residues
    if pdb in pdb_info and len(residues) <= len(pdb_info[pdb]["residues"]):
        continue

    # check resolution
    content = requests.get(rcsb_structure + pdb).content.decode('utf-8')
    content = content.replace("&nbsp", " ").split("Resolution: </strong>")

    # drop if fail to get resolution
    if len(content) == 1:
        continue

    resolution = float(content[1][:5])
    # drop bad resolution
    if resolution > 3:
        continue

    # save PDB
    req = requests.get(rcsb_download + pdb + ".pdb")
    pdb_dir = f"../data/pdbs/{pdb}.pdb"
    with open(pdb_dir, "wb") as f:
        f.write(req.content)

    # extract one-letter sequence
    sequence = extract_sequence(pdb_dir, chain)
    # drop PDBs with short sequence (abnormal)
    if len(sequence) <= 10:
        continue
    # drop if cannot understand sequence, all '-'
    if len(set(sequence)) == 1:
        continue

    # drop if modulator not in PDB
    exist = False
    pdb_file: List[str] = []
    with open(pdb_dir, "r") as f:
        f.readlines()
    for pdb_line in pdb_file:
        if (
            pdb_line[:6] == "HETATM" and
            modulator == pdb_line[17:20].strip() and
            mod_id == pdb_line[22:26].strip() and
            chain == pdb_line[21]
        ):
            exist = True
            break
    if not exist:
        continue

    # save entry
    pdb_info[pdb] = {
        "modulator": modulator,
        "mod_id": mod_id,
        "chain": chain,
        "residues": residues,
        "sequence": sequence
    }


print("Number of PDBs: ", len(pdb_info))
with open("../data/clean_data/pdb_info.pkl", "wb") as f:
    pickle.dump(pdb_info, f)

# detect pockets
for pdb in list(pdb_info.keys()):
    chain = pdb_info[pdb]["chain"]
    os.system(f"fpocket -f ../data/pdbs/{pdb}.pdb -k {chain}")
    os.system(f"mv ../data/pdbs/{pdb}_out ../data/pockets/")
