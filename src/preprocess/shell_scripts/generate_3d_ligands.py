# shell script to generate 3d ligands with openbabel
# Open Babel is a chemical toolbox designed to speak the many languages of chemical data.
# Further Reference: http://openbabel.org/wiki/Main_Page
# Note: openbabel needs python < 3.7 to run. I'm running this under python 3.6.
# Note: use openbabel version <= 2.4.0 if possible, the higher versions might have trouble
# generating 3d coordinates sometimes, don't really know why.

r"""
Example Run Output:
Open Babel 3.1.0 -- Sep 16 2021 -- 06:56:50
Converting 3431 smiles to pdbs ................................... done.
Time Out LIDs: ['270', '408', '758', '1087', '1374', '2004', '2111', '2246', '2437', '2831']
"""

import subprocess
import pandas as pd
from src.global_vars import *


if __name__ == '__main__':

    # read ligand.csv and convert to pdbs
    df = pd.read_csv(LIGAND_DIR)
    subprocess.run("obabel -V", shell=True)
    print("Converting {} smiles to pdbs ".format(len(df)), end='', flush=True)

    timeout_lids = []
    for i in range(len(df)):
        if i % 100 == 0:
            print(".", end='', flush=True)  # progress indicator
        lid = str(df.LID[i])
        smiles = r'{}'.format(df.Smiles[i])
        log_file = LIGAND_PDB_DIR + "/" + "log.txt"
        err_file = LIGAND_PDB_DIR + "/" + "errors.txt"
        try:
            subprocess.run("obabel -ismi -:'{}' -opdb -O {}.pdb --gen3d best --errorlevel 1 >{} 2>{}"
                           .format(smiles, LIGAND_PDB_DIR + "/" + lid, log_file, err_file),
                           shell=True, timeout=180)
        except subprocess.TimeoutExpired:  # timeout after 300 secs
            try:
                subprocess.run("obabel -ismi -:'{}' -o pdb -O {}.pdb --gen3d fast --errorlevel 1 >{} 2>{}"
                               .format(smiles, LIGAND_PDB_DIR + "/" + lid, log_file, err_file),
                               shell=True, timeout=300)  # run with fast 3d generation instead
            except subprocess.TimeoutExpired:
                timeout_lids.append(lid)
                subprocess.run("obabel -ismi -:'{}' -o pdb -O {}.pdb --errorlevel 1 >{} 2>{}"
                               .format(smiles, LIGAND_PDB_DIR + "/" + lid, log_file, err_file),
                               shell=True)  # run without 3d generation instead

    print(" done.")
    print('Time Out LIDs: {}'.format(timeout_lids))
