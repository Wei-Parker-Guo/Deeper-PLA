# this file just defines the global variable we are gonna use in the project

# directories
CENTROIDS_DIR = '../data/centroids.csv'
PROTEIN_PDBS_DIR = '../data/pdbs'
LIGAND_DIR = '../data/ligand.csv'
LIGAND_PDB_DIR = '../data/ligand-pdbs'
GT_PAIR_DIR = '../data/pair.csv'

# definitions
HYDROPHOBIC = 0  # hydrophobic atom code
POLAR = 1  # polar atom code
HYDROPHOBIC_ATOMS = {'C'}
POLAR_ATOMS = {'O', 'N'}
