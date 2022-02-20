# This script tries to fix the 3d ligands not properly generated by openbabel.
# These ligand pdbs have all zero coordinates on each atom.
# The results are often due to molecule structures being too complex.
# We will try to fix them using rdkit.
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToPDBFile
from src.global_vars import LIGAND_DIR, LIGAND_PDB_DIR
from src.preprocess.read_pdb_file import read_ligand_pdb


if __name__ == '__main__':
    print("Reading ligands ... ")
    df = pd.read_csv(LIGAND_DIR)
    illegal_3d_ligands = []
    for i in range(len(df)):
        lid = df.LID[i]
        smiles = str(df.Smiles[i])

        illegal, atom_types, connections, xs, ys, zs = read_ligand_pdb(
            '{}/{}.pdb'.format(LIGAND_PDB_DIR, lid))

        if illegal:
            illegal_3d_ligands.append(lid)

    print("Found {} illegal 3D ligands:\n{}\n".format(
        len(illegal_3d_ligands), illegal_3d_ligands))

    print("Trying to fix them using RdKit ... ", end='')
    for lid in illegal_3d_ligands:
        smiles = str(df.Smiles[lid])
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            MolToPDBFile(mol, '{}/{}.pdb'.format(LIGAND_PDB_DIR, lid))
            illegal_3d_ligands.remove(lid)  # remove if succeed in fix
        except:
            continue

    print('The following ligands are not fixed:\n{}'.format(illegal_3d_ligands))