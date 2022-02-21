# this file just defines the global variable we are going to use in the project
# it also includes all the hyper parameters modifiable for the model.

# directories
CENTROIDS_DIR = '../data/centroids.csv'
PROTEIN_PDBS_DIR = '../data/pdbs'
LIGAND_DIR = '../data/ligand.csv'
LIGAND_PDB_DIR = '../data/ligand-pdbs'
GT_PAIR_DIR = '../data/pair.csv'
MOL2VEC_MODEL_DIR = '../models/mol2vec/model_300dim.pkl'

# BindGrid definitions
HYDROPHOBIC = 0  # hydrophobic atom code
POLAR = 1  # polar atom code
HYDROPHOBIC_ATOMS = {'C'}
POLAR_ATOMS = {'O', 'N'}
AUGMENT_ROTATION = 2  # augment data by rotating the grid for another n times
# the atom types are derived from the Arpeggio paper for interactions of interest
# these dicts record their names and channel number
# see reference in README.md
PROTEIN_ATOMS = {'C': 0, 'N': 1, 'O': 2}
LIGAND_ATOMS = {'C': 3, 'N': 4, 'O': 5, 'P': 6, 'S': 7, 'F': 8, 'Cl': 9, 'Br': 10, 'I': 11}
PROTEIN_VOLUME_CH = 12
LIGAND_VOLUME_CH = 12
VDW_Radius = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'P': 1.80, 'S': 1.80, 'F': 1.47,
              'Cl': 1.75, 'Br': 1.83, 'I': 1.98, 'H': 1.10}  # Van der Waals radius in atomic radius
MAX_LIGAND_R = 36  # max ligand radius for defining a bind grid
GRID_CHANNELS = 14

# Mol2Vec embedding definitions
EMBED_DIM = 76
EMBED_PADDING_DIM = 76  # we pad the embeddings if they don't have enough components

# Network Architecture Definitions
SHUFFLE_G = 4  # number of shuffle groups in each shuffle group unit
SHUFFLE_CHS = [MAX_LIGAND_R, 244, 488, 976]
SHUFFLE_UNITS = [2, 3, 3]
REG_CH = 1024  # channel number for global PW conv at regression block
SMILES_CNN_CH = [1, 64, 128, 256]
SMILES_CNN_G = [3, 3, 3]
