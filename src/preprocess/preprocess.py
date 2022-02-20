import subprocess

import pandas as pd
import torch

from src.preprocess.read_pdb_file import read_pdb, read_ligand_pdb
from src.utilities import print_v
from src.global_vars import *
from src.preprocess.data_structures import Protein, Ligand
from src.preprocess.bind_grid import BindGrid
from src.preprocess.smiles2vec import Smiles2Vec


# preprocessor for training
class TrainPreprocessor:
    def __init__(self, smiles_dim=EMBED_DIM, device='cpu', verbose=False):
        self.device = device
        self.verbose = verbose
        self.centroids, self.ligands, self.gt_pairs, self.proteins = {}, {}, {}, {}
        self.illegal_3d_ligands = []
        self.read_objects()
        self.smile2vec = Smiles2Vec(smiles_dim)

    # read the objects we are interested in
    def read_objects(self):
        # read pair.csv
        print_v("Reading protein ligand pairs ... ", self.verbose)
        df = pd.read_csv(GT_PAIR_DIR)
        for i in range(len(df)):
            self.gt_pairs[str(df.PID[i])] = (str(df.LID[i]))
        print_v("done.\n", self.verbose)

        # read centroids.csv
        print_v("Reading centroids ... ", self.verbose)
        df = pd.read_csv(CENTROIDS_DIR)
        for i in range(len(df)):
            self.centroids[str(df.PID[i])] = (float(df.x[i]), float(df.y[i]), float(df.z[i]))
        print_v("done.\n", self.verbose)

        # read proteins
        print_v("Reading proteins ... ", self.verbose)
        for pid in self.centroids.keys():
            xs, ys, zs, atom_types = read_pdb("{}/{}.pdb".format(PROTEIN_PDBS_DIR, pid))
            protein = Protein(atom_types, xs, ys, zs)
            self.proteins[pid] = protein
        print_v("done.\n", self.verbose)

        # read ligands
        print_v("Reading ligands ... ", self.verbose)
        df = pd.read_csv(LIGAND_DIR)
        for i in range(len(df)):
            lid = str(df.LID[i])
            smiles = str(df.Smiles[i])

            illegal, atom_types, connections, xs, ys, zs = read_ligand_pdb(
                '{}/{}.pdb'.format(LIGAND_PDB_DIR, lid))

            if illegal:
                self.illegal_3d_ligands.append(lid)
            else:
                self.ligands[lid] = Ligand(smiles, atom_types, connections, xs, ys, zs)
        print_v("Found {} illegal 3D ligands:\n{}\n".format(
            len(self.illegal_3d_ligands), self.illegal_3d_ligands), self.verbose)
        print_v("done.\n", self.verbose)

    # generate bindgrids as tensors given a protein and a batch of ligands
    def generate_bindgrids(self, pid, centroid, lids, smiles):
        bindgrids = []
        for i in range(len(lids)):
            lid = lids[i]
            # if ligand is not in generated pdbs, try creating a new one
            if lid not in self.ligands:
                log_file = LIGAND_PDB_DIR + "/" + "log.txt"
                err_file = LIGAND_PDB_DIR + "/" + "errors.txt"
                try:
                    subprocess.run("obabel -ismi -:'{}' -o pdb -O {}.pdb --gen3d >{} 2>{}"
                                   .format(smiles[i], LIGAND_PDB_DIR + "/" + lid, log_file, err_file),
                                   shell=True, timeout=300)
                except subprocess.TimeoutExpired:  # timeout after 300 secs
                    subprocess.run("obabel -ismi -:'{}' -o pdb -O {}.pdb >{} 2>{}"
                                   .format(smiles[i], LIGAND_PDB_DIR + "/" + lid, log_file, err_file),
                                   shell=True)  # run without 3d generation instead

                # read the newly made pdb file
                illegal, atom_types, connections, xs, ys, zs = read_ligand_pdb(
                    '{}/{}.pdb'.format(LIGAND_PDB_DIR, lid))

                if illegal:
                    self.illegal_3d_ligands.append(lid)
                else:
                    self.ligands[lid] = Ligand(smiles, atom_types, connections, xs, ys, zs)

            bindgrids.append(BindGrid(self.proteins[pid], self.ligands[lid], centroid).grid)
        return bindgrids

    # generate embeddings as tensors for a batch of ligand smiles
    def generate_embeddings(self, smiles):
        return self.smile2vec.get_embeddings(smiles)

    # preprocess all the data in this object to feed the model
    def preprocess(self, PID, centroid, LID, ligands):
        pass


# Preprocessor for inference/test
class TestPreprocessor(TrainPreprocessor):
    def __init__(self, smiles_dim=64, device='cpu', verbose=False):
        super().__init__(smiles_dim, device, verbose)

    def preprocess(self, PID, centroid, LID, ligands):
        data = []
        return data


if __name__ == '__main__':
    torch.set_printoptions(profile='full')

    # construction and reading
    train_processor = TrainPreprocessor(verbose=True)

    print("\nSample Protein:")
    print(train_processor.proteins['1A0Q'].atom_types[:5])
    print(train_processor.proteins['1A0Q'].xs[:5])
    print(train_processor.proteins['1A0Q'].ys[:5])
    print(train_processor.proteins['1A0Q'].zs[:5])

    # find max protein r
    max_protein_r = 0
    for protein in train_processor.proteins.values():
        max_r = max(max(protein.xs), max(protein.ys), max(protein.zs))
        if max_r > max_protein_r:
            max_protein_r = max_r
    print("Max Protein radius: {}".format(max_protein_r))

    print("\nSample Ligand:")
    print(train_processor.ligands['1'].smiles)
    print(train_processor.ligands['1'].atom_types[:5])
    print(list(train_processor.ligands['1'].connections.items())[:5])
    print(train_processor.ligands['1'].xs[:5])
    print(train_processor.ligands['1'].ys[:5])
    print(train_processor.ligands['1'].zs[:5])

    # find max ligand r
    max_ligand_r = 0
    for ligand in train_processor.ligands.values():
        max_r = max(max(ligand.xs), max(ligand.ys), max(ligand.zs))
        if max_r > max_ligand_r:
            prev_ligand_r = max_ligand_r
            max_ligand_r = max_r
    print("Max Ligand radius: {}".format(max_ligand_r))

    # preprocessing

    # building a sample grid
    print("\nBuilding a sample bind grid:")
    bind_grid = BindGrid(train_processor.proteins['1A0Q'],
                         train_processor.ligands['1'], train_processor.centroids['1A0Q'])
    print(bind_grid.grid.shape)
    # augmenting a sample grid
    print("\nAugmenting the sample bind grid:")
    bind_grid.rotation_augment()
    print(bind_grid.augmented_grids.shape)

    # building sample smiles embeddings
    print("\nGenerating 5 sample smiles embeddings with dim 64:")
    embeds = train_processor.generate_embeddings(
        [v.smiles for (_, v) in list(train_processor.ligands.items())[:5]])
    print(embeds.shape)
