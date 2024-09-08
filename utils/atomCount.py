#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 06/13/2021
# Author: Sian Xiao & Hao Tian

"""
Two examples for function atomCount
1PJ3:
    fileDirection   = '../data/pockets/1PJ3_out/pockets/pocket1_atm.pdb'
    info            = 'Chain A:GLN64,ARG67,ILE88,ARG91,LEU95; Chain B:PHE1127,ARG1128'
    return          = 10
1AO0:
    fileDirection   = '../data/pockets/1AO0_out/pockets/pocket1_atm.pdb'
    info            = 'Chain A:HIS25,TYR242,SER244,ARG245,PRO246,ARG259,PRO281,ASP282,SER283,LYS305,LYS328; \
        Chain B:ILE304,LYS305,ASN306,ARG307'
    return          = 31
"""


def atomCount(fileDirection: str, info: str) -> int:
    """compare how many matched heavy atoms

    Args:
        fileDirection (str): file location '../data/pockets/{pdb}_out/pockets/*.pdb'
        info (str): correspoding allosteric info from ASD

    Returns:
        int: how many matched heavy atoms
    """

    # collect allosteric info
    atomTarget = dict()
    info = info.split("\t")[-1].strip().split(";")  # 'allosteric_site_residue'

    for chains in info:
        chains = chains.strip()
        chainID = chains[6]
        atoms = chains[8:].split(",")
        # map atoms to chain ID
        for atom in atoms:
            atomTarget[atom] = chainID

    # count matched atoms
    pocket = open(fileDirection, "r").readlines()
    count = 0
    for line in pocket:
        if line.startswith("ATOM"):
            # Chain identifier
            chainID = line[21]
            # Residue name and Residue sequence number
            atom = line[17:20] + line[22:26].strip()
            # same atom and same chain ID
            if atom in atomTarget and atomTarget[atom] == chainID:
                count += 1
    return count

