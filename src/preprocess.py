import pandas as pd
from typing import NamedTuple  # requires python version > 3.6
from read_pdb_file import read_pdb
from global_vars import *
from utilities import print_v


class Protein(NamedTuple):
    atom_types: list
    xs: list
    ys: list
    zs: list


class Ligand(NamedTuple):
    smiles: str
    atom_types: list
    connections: dict
    xs: list
    ys: list
    zs: list


# preprocessor for training
class TrainPreprocessor:
    def __init__(self, device='cpu', verbose=False):
        self.device = device
        self.verbose = verbose
        self.centroids, self.ligands, self.gt_pairs, self.proteins = {}, {}, {}, {}
        self.illegal_3d_ligands = []
        self.read_objects()

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

            #read 3d structure
            with open("{}/{}.pdb".format(LIGAND_PDB_DIR, lid), 'r') as file:
                strline_L = file.readlines()
            file.close()
            strline_L = [strline.strip() for strline in strline_L]

            atom_types, xs, ys, zs = [], [], [], []
            connections = {}
            zero_xyz = ('0.000', '0.000', '0.000')
            illegal = True

            for strline in strline_L:
                tokens = strline.split()
                if tokens[0] == "HETATM" or tokens[0] == "ATOM":
                    atom_types.append(tokens[2])
                    xs.append(tokens[5])
                    ys.append(tokens[6])
                    zs.append(tokens[7])
                    cur_xyz = (xs[-1], ys[-1], zs[-1])
                    if illegal and cur_xyz != zero_xyz:  # found illgal 3d structure (duplicated coords)
                        illegal = False
                elif tokens[0] == "CONECT":
                    connections[tokens[1]] = [t for t in tokens[2:]]

            if illegal:
                self.illegal_3d_ligands.append(lid)
            else:
                self.ligands[lid] = Ligand(smiles, atom_types, connections, xs, ys, zs)

        print_v("Found {} illegal 3D ligands:\n{}\n".format(
            len(self.illegal_3d_ligands), self.illegal_3d_ligands), self.verbose)
        print_v("done.\n", self.verbose)

    # preprocess all the data in this object to feed the model
    def preprocess(self):
        pass


# Preprocessor for inference/test
class TestPreprocessor(TrainPreprocessor):
    def __init__(self, device='cpu', verbose=False):
        super().__init__(device, verbose)

    def preprocess(self, PID, centroid, LID, ligands):
        data = []
        return data


if __name__ == '__main__':
    # construction and reading
    train_processor = TrainPreprocessor(verbose=True)

    print("\nSample Protein:")
    print(train_processor.proteins['1A0Q'].atom_types[:5])
    print(train_processor.proteins['1A0Q'].xs[:5])
    print(train_processor.proteins['1A0Q'].ys[:5])
    print(train_processor.proteins['1A0Q'].zs[:5])

    print("\nSample Ligand:")
    print(train_processor.ligands['1'].smiles)
    print(train_processor.ligands['1'].atom_types[:5])
    print(list(train_processor.ligands['1'].connections.items())[:5])
    print(train_processor.ligands['1'].xs[:5])
    print(train_processor.ligands['1'].ys[:5])
    print(train_processor.ligands['1'].zs[:5])

    # preprocessing
