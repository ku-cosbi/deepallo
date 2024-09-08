# import requests, pickle, os
# from Bio.PDB import PDBParser

# def get_sifts_mapping(pdb_id, chain_id):
#     sifts_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/{pdb_id.lower()}"
#     response = requests.get(sifts_url)

#     if response.status_code != 200:
#         print("Error: Unable to retrieve SIFTS data.")
#         return None

#     data = response.json()

#     if pdb_id not in data:
#         print("Error: PDB ID not found in SIFTS data.")
#         return None

#     mappings = data[pdb_id]["UniProt"]

#     for uniprot_id, mapping_data in mappings.items():
#         for mapping in mapping_data["mappings"]:
#             if mapping["chain_id"] == chain_id:
#                 pdb_start, pdb_end = mapping["start"]["residue_number"], mapping["end"]["residue_number"]
#                 uniprot_start, uniprot_end = mapping["unp_start"], mapping["unp_end"]
#                 pdb_to_uniprot_mapping = {
#                     pdb_residue: uniprot_start + (pdb_residue - pdb_start)
#                     for pdb_residue in range(pdb_start, pdb_end + 1)
#                 }
#                 return uniprot_id, pdb_to_uniprot_mapping

#     print("Error: Chain not found in SIFTS data.")
#     return None

# # Example usage:
# pdb_id = "3f1o"
# chain_id = "A"
# uniprot_id, pdb_to_uniprot_mapping = get_sifts_mapping(pdb_id, chain_id)

# print(f"UniProt ID: {uniprot_id}")
# print("PDB to UniProt residue mapping:")
# print(pdb_to_uniprot_mapping)

##################################################################################################3

# import json
# import io
# import requests
# from Bio.PDB import MMCIFParser
# from Bio.PDB.MMCIF2Dict import MMCIF2Dict

# # Download the mmCIF file
# pdb_id = "1A3W"
# mmcif_url = f"https://files.rcsb.org/download/{pdb_id.lower()}.cif"
# response = requests.get(mmcif_url)

# if response.status_code != 200:
#     print("Error: Unable to download the mmCIF file.")
# else:
#     mmcif_file = io.StringIO(response.text)

#     # Parse the mmCIF file
#     mmcif_dict = MMCIF2Dict(mmcif_file)

#     # Get the RAF data
#     raf_key = "_pdbx_poly_seq_scheme.auth_seq_num"
#     raf_data = mmcif_dict[raf_key]

#     # Get the chain data
#     chain_key = "_pdbx_poly_seq_scheme.pdb_strand_id"
#     chain_data = mmcif_dict[chain_key]

#     # Get the sequence indices
#     sequence_key = "_pdbx_poly_seq_scheme.seq_id"
#     sequence_data = mmcif_dict[sequence_key]

#     # Filter for the corresponding chain
#     chain_id = "A"
#     filtered_raf_data = [[atom_residue_number, int(sequence_id) - 1] for atom_residue_number, sequence_id, chain in zip(raf_data, sequence_data, chain_data) if chain == chain_id]

#     # Map SEQRES residues to ATOM indices
#     seqres_to_atom_mapping = {}
#     # seqres_index = 1

#     for raf_index_data in filtered_raf_data:
#         atom_residue_number = raf_index_data[0]
#         sequence_index = raf_index_data[1]
#         if atom_residue_number != "?":
#             seqres_to_atom_mapping[atom_residue_number] = sequence_index

#     print(seqres_to_atom_mapping)


##################################################################################################3

import pickle
from tqdm import tqdm
from Bio.PDB import PDBParser
from utils.pocket_coordinates import pocket_coordinates


pdb_diverse = pickle.load(open("../data/clean_data/pdb_diverse.pkl", "rb"))
seq_annotated = pickle.load(open("../data/classification/seq_annotated.pkl", "rb"))
LM_RESIDUE_PATH = "../data/classification/LM_residue_partitioned"
pdb_train = pickle.load(open(f"{LM_RESIDUE_PATH}/LM_residue_test.pkl", "rb"))
updated_pdb_train = []

for pdb in tqdm(pdb_train):
    for seq in seq_annotated:
        if pdb["pdb"] == seq["pdb"]:
            updated_pdb_train.append({ **pdb, "chain": seq["chain"],  "pocket_coordinates":seq["pocket_coordinates"] })
            break

pickle.dump(updated_pdb_train, open(f"{LM_RESIDUE_PATH}/LM_residue_test_up.pkl", "wb"))

# for pdb in tqdm(seq_annotated):
#     # Load the structure
#     protein_name = pdb["pdb"]
#     chain_id = pdb_diverse[pdb["pdb"]]["chain"]

#     pocket_coord = pocket_coordinates(f"../data/pdbs/{protein_name}.pdb", f"../data/pockets/{protein_name}_out/pockets/", protein_name, chain_id, pdb["poc_indices"])
#     updated_seq_annotated.append({ **pdb, "chain": chain_id, "pocket_coordinates": pocket_coord })

# pickle.dump(updated_seq_annotated, open("../data/classification/seq_annotated_up.pkl", "wb"))
