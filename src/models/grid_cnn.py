# this module constructs the entire model pipeline for forwarding bind grids
import torch
from torch import nn
from src.global_vars import *


class ShuffleGroup(nn.Module):
    def __init__(self, device='cpu'):
        super(ShuffleGroup, self).__init__()
        self.device = device
        self.model = nn.ModuleDict({
            
        })

class GridCNN(nn.Module):
    def __init__(self, device='cpu'):
        super(GridCNN, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            # atom info integration block
            nn.Conv3d(GRID_CHANNELS, MAX_LIGAND_R, (1, 1, 1), (1, 1, 1), device=self.device),
            nn.MaxPool3d((3, 3, 3), (1, 1, 1)),
            # feature extraction block

        )

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)
