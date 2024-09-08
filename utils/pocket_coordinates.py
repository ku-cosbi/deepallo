import glob
import numpy as np
from Bio.PDB import PDBParser

class BreakIt(Exception): pass

def pocket_coordinates(pdb_path: str, pocket_dir: str, pdb_id: str, chain_id: str, pockets_indices: list):
    parser = PDBParser(QUIET=True)

    # Load the structure
    structure = parser.get_structure(pdb_id, pdb_path)

    # Get the model
    model = structure[0]

    # Get the chain
    chain = model[chain_id]

    pocket_coordinates = []
    for pockets in pockets_indices:
        cur_pocket_coordinates = []
        for res_idx in pockets:
            try:
                cur_pocket_coordinates.append(chain[int(res_idx)]['CA'].get_coord())
            except Exception as e:
                try:
                    pocket_names = glob.glob(pocket_dir + "*.pdb")
                    pocket_names = sorted(
                        pocket_names,
                        key=lambda x: int(x.split("pocket")[-1].split("_")[0])
                    )
                    for pocket_name in pocket_names:
                        pocket = None
                        with open(pocket_name, "r") as f:
                            pocket = f.readlines()
                        for line in pocket:
                            if line[:4] == "ATOM" and line[21] == chain_id and line[22:26].strip() == res_idx:
                                cur_pocket_coordinates.append(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype='float32'))
                                raise BreakIt
                except BreakIt:
                    pass

        pocket_coordinates.append(cur_pocket_coordinates)

    return pocket_coordinates
