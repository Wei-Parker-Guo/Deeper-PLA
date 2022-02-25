# file for generating datasets with multi threading
import torch

# define a custom dataset format for our data
import random
import pandas as pd
from typing import NamedTuple
from torch.utils.data import Dataset
from src.preprocess.bind_grid import BindGrid
from src.global_vars import AUGMENT_ROTATION


class DataItem(NamedTuple):
    grid: torch.Tensor
    embed: torch.Tensor
    label: float


class ProteinLigandDataset(Dataset):
    def __init__(self, pairs, train_processor, batch_size, rot_aug=True):
        self.pairs = pairs  # a list of pairs from querying pair.csv
        self.batch_size = batch_size
        self.train_processor = train_processor
        self.rot_aug = rot_aug  # augment data by random grid rotation
        self.data = []
        self.gen_data()

    # based on the given pairs, generate the full dataset with both true and false pairs
    def gen_data(self):
        smiles = []
        grids = []
        labels = []
        for pair in self.pairs:
            pid, lid = pair
            for i in range(self.batch_size):
                lid = int(lid)
                if i != 0:  # generate false pair
                    prev_lid = lid
                    while prev_lid == lid or (str(lid) not in self.train_processor.gt_pairs.values()):
                        lid = random.randint(1, len(self.train_processor.ligands))
                lid = str(lid)
                grid = BindGrid(self.train_processor.proteins[pid], self.train_processor.ligands[lid],
                                self.train_processor.centroids[pid])
                gs = [grid.grid]
                smiles.append(self.train_processor.ligands[lid].smiles)
                if self.rot_aug:
                    gs = grid.rotation_augment()
                for j in range(len(gs)):
                    grids.append(gs[j])
                    if i == 0:
                        labels.append(1.0)
                    else:
                        labels.append(0.0)

        embeds = self.train_processor.generate_embeddings(smiles)
        for i in range(len(labels)):
            ie = i // (AUGMENT_ROTATION + 1)
            ie = ie if ie < len(embeds) else len(embeds) - 1
            self.data.append(DataItem(grids[i], embeds[ie], labels[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        i = self.data[idx]
        return i.grid, i.embed, i.label


def gen_dataset(pairs, cache_dir, index, train_preprocessor,
                rot_aug=True, batch_size=2, cache_on_disk=True):
    dataset = ProteinLigandDataset(pairs, train_preprocessor, batch_size, rot_aug)
    torch.save(dataset, '{}/{}.data'.format(cache_dir, index))
