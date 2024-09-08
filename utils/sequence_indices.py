import io
import requests
from collections import defaultdict
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

def sequence_indices(pdb_id: str, chain_id: str):
    # Download the mmCIF file
    mmcif_url = f"https://files.rcsb.org/download/{pdb_id.lower()}.cif"
    response = requests.get(mmcif_url)

    if response.status_code != 200:
        print("Error: Unable to download the mmCIF file.")
    else:
        mmcif_file = io.StringIO(response.text)

        # Parse the mmCIF file
        mmcif_dict = MMCIF2Dict(mmcif_file)

        # Get the RAF data
        raf_key = "_pdbx_poly_seq_scheme.pdb_seq_num"
        raf_data = mmcif_dict[raf_key]

        # Get the chain data
        chain_key = "_pdbx_poly_seq_scheme.pdb_strand_id"
        chain_data = mmcif_dict[chain_key]

        # Get the sequence indices
        sequence_key = "_pdbx_poly_seq_scheme.seq_id"
        sequence_data = mmcif_dict[sequence_key]

        # Filter for the corresponding chain
        filtered_raf_data = [[atom_residue_number, int(sequence_id) - 1] for atom_residue_number, sequence_id, chain in zip(raf_data, sequence_data, chain_data) if chain == chain_id]

        # Map SEQRES residues to ATOM indices
        seqres_to_atom_mapping = {}
        # seqres_index = 1

        for raf_index_data in filtered_raf_data:
            atom_residue_number = raf_index_data[0]
            sequence_index = raf_index_data[1]
            if atom_residue_number != "?":
                seqres_to_atom_mapping[atom_residue_number] = sequence_index

    return seqres_to_atom_mapping