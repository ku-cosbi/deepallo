#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil


# init pdbs folder
shutil.rmtree("../data/pdbs/", ignore_errors=True)
os.system("mkdir ../data/pdbs/")

# init pockets folder
shutil.rmtree("../data/pockets/", ignore_errors=True)
os.system("mkdir ../data/pockets/")

# init classification folder
shutil.rmtree("../data/classification/", ignore_errors=True)
os.system("mkdir ../data/classification/")

# init clean_data folder
shutil.rmtree("../data/clean_data/", ignore_errors=True)
os.system("mkdir ../data/clean_data/")
