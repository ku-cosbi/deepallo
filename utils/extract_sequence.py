#!/usr/bin/env python
# -*- coding: utf-8 -*-

THREE_ONE_DIGIT = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "ASX": "B",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLX": "Z",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    'TPO': 'T',
    'SEP': 'S',
    'LLP': 'K',
    'CSO': 'C',
    'CME': 'C',
    'KCX': 'K',
    'YCM': 'C'
}


def extract_sequence(pdb_path, chain):
    lines = open(pdb_path, "r").readlines()
    res_one_letters = ""
    for line in lines:
        if line[:6] == "SEQRES" and line[11] == chain:
            residues = line[19:].strip().split(" ")
            res_one_letter = [*map(THREE_ONE_DIGIT.get, residues)]
            res_one_letter = [
                char if char is not None else "-" for char in res_one_letter
            ]
            res_one_letters += "".join(res_one_letter)
    return res_one_letters
