import math
import random

import torch
from src.preprocess.data_structures import Protein, Ligand
from src.global_vars import *


# PCMax algorithm to convert atom to voxel representation
def pcmax(r_vdw, r):
    return 1 - math.exp(-((r_vdw / r) ** 12))


# Data structure representing a 3D binding grid of ligand and local protein residues
# can output a tensor of its representation
class BindGrid:
    def __init__(self, protein: Protein, ligand: Ligand, binding_loc):
        self.grid = torch.zeros(size=(GRID_CHANNELS, MAX_LIGAND_R, MAX_LIGAND_R, MAX_LIGAND_R))
        self.augmented_grids = torch.Tensor()
        self.binding_loc = binding_loc
        self.cache = {}  # cache repetitive calculations
        self.create_cache()
        self.create_tensor(protein, ligand)

    def create_cache(self):
        self.cache['lbx'] = self.binding_loc[0] - MAX_LIGAND_R // 2
        self.cache['hbx'] = self.binding_loc[0] + MAX_LIGAND_R // 2
        self.cache['lby'] = self.binding_loc[1] - MAX_LIGAND_R // 2
        self.cache['hby'] = self.binding_loc[1] + MAX_LIGAND_R // 2
        self.cache['lbz'] = self.binding_loc[2] - MAX_LIGAND_R // 2
        self.cache['hbz'] = self.binding_loc[2] + MAX_LIGAND_R // 2

    def create_tensor(self, protein, ligand):
        # protein
        for i in range(len(protein.atom_types)):
            atom_t = protein.atom_types[i]
            if atom_t in PROTEIN_ATOMS:
                x, y, z = protein.xs[i], protein.ys[i], protein.zs[i]
                # record occupancy channel
                self.record_occupancy(x, y, z, VDW_Radius[atom_t], PROTEIN_VOLUME_CH, False)
                # record atom channel
                atom_c = PROTEIN_ATOMS[atom_t]  # atom channel number
                r_vdw = VDW_Radius[atom_t]
                self.record_occupancy(x, y, z, r_vdw, atom_c, False)
        # ligand
        for i in range(len(ligand.atom_types)):
            atom_t = ligand.atom_types[i]
            if atom_t in LIGAND_ATOMS:
                x, y, z = ligand.xs[i], ligand.ys[i], ligand.zs[i]
                # record occupancy channel
                self.record_occupancy(x, y, z, VDW_Radius[atom_t], LIGAND_VOLUME_CH, True)
                # record atom channel
                atom_c = LIGAND_ATOMS[atom_t]  # atom channel number
                r_vdw = VDW_Radius[atom_t]
                self.record_occupancy(x, y, z, r_vdw, atom_c, False)

    def record_occupancy(self, x, y, z, r_vdw, channel, is_ligand):
        # only record local structure within the grid
        bias = self.binding_loc if is_ligand else (0.0, 0.0, 0.0)
        if not (self.cache['lbx'] <= bias[0] + x <= self.cache['hbx'] and self.cache['lby'] <= bias[1] + y <= self.cache['hby']
                and self.cache['lbz'] <= bias[2] + z <= self.cache['hbz']):
            return
        # each atom can contribute to voxels double of its van der Waals radius
        lx, ux = bias[0] + x - 2 * r_vdw, bias[0] + x + 2 * r_vdw
        lx = lx if lx >= self.cache['lbx'] else self.cache['lbx']
        ux = ux if ux <= self.cache['hbx'] else self.cache['hbx']
        ly, uy = bias[1] + y - 2 * r_vdw, bias[1] + y + 2 * r_vdw
        ly = ly if ly >= self.cache['lby'] else self.cache['lby']
        uy = uy if uy <= self.cache['hby'] else self.cache['hby']
        lz, uz = bias[2] + z - 2 * r_vdw, bias[2] + z + 2 * r_vdw
        lz = lz if lz >= self.cache['lbz'] else self.cache['lbz']
        uz = uz if uz <= self.cache['hbz'] else self.cache['hbz']
        for i in range(int(math.ceil(lx)), int(math.floor(ux)) + 1):
            for j in range(int(math.ceil(ly)), int(math.floor(uy)) + 1):
                for k in range(int(math.ceil(lz)), int(math.floor(uz)) + 1):
                    contrib = pcmax(r_vdw, math.sqrt(
                        (i + 0.5 - x) ** 2 + (j + 0.5 - y) ** 2 + (k + 0.5 - z) ** 2))
                    ix = int(i - self.binding_loc[0] + MAX_LIGAND_R // 2)
                    iy = int(j - self.binding_loc[1] + MAX_LIGAND_R // 2)
                    iz = int(k - self.binding_loc[2] + MAX_LIGAND_R // 2)
                    self.grid[channel][ix][iy][iz] += contrib

    # data augmentation by random 90x degree rotation, return a batch of augmented grids as tensor
    def rotation_augment(self):
        grids = [self.grid]
        for i in range(AUGMENT_ROTATION):
            r = torch.rot90(self.grid, random.randint(0, i + 1), [1, 2])
            r = torch.rot90(r, random.randint(0, i + 1), [2, 3])
            r = torch.rot90(r, random.randint(0, i + 1), [3, 1])
            grids.append(r)
        self.augmented_grids = torch.stack(grids)
        return self.augmented_grids
