# this module constructs the entire model pipeline for forwarding bind grids
from collections import OrderedDict

import torch
from torch import nn
from torchsummary import summary

from src.global_vars import *


class BasicUnit(nn.Module):
    def __init__(self, out_channel, device='cpu'):
        super(BasicUnit, self).__init__()
        self.device = device
        self.out_channel = out_channel
        # we only need branch 2 here since branch 1 is identity
        oc = out_channel // 2
        self.branch2 = nn.Sequential(
            # PW
            nn.Conv3d(oc, oc, (1, 1, 1), (1, 1, 1), device=self.device),
            nn.BatchNorm3d(oc, device=self.device),
            nn.LeakyReLU(),
            # DW
            nn.Conv3d(oc, oc, (3, 3, 3), (1, 1, 1), padding=1, device=self.device),
            nn.BatchNorm3d(oc, device=self.device),
            # PW
            nn.Conv3d(oc, oc, (1, 1, 1), (1, 1, 1), device=self.device),
            nn.BatchNorm3d(oc, device=self.device),
            nn.LeakyReLU()
        )

    def forward(self, x):
        b1, b2 = torch.split(x, self.out_channel // 2, dim=1)
        # feed b2 to branch 2
        b2 = self.branch2.forward(b2)
        # concat, shuffle and return
        x = torch.cat((b1, b2), dim=1)
        N, C, H, W, D = x.shape
        g = SHUFFLE_G
        return x.view(N, g, C // g, H, W, D).permute(0, 2, 1, 3, 4, 5).reshape(N, C, H, W, D)


class DownSampleUnit(nn.Module):
    def __init__(self, in_channel, out_channel, device='cpu'):
        super(DownSampleUnit, self).__init__()
        self.device = device
        oc = out_channel // 2
        # branch 1 is the identity down sample branch and 2 the extraction branch
        self.branch1 = nn.Sequential(
            # DW
            nn.Conv3d(in_channel, oc, (3, 3, 3), (2, 2, 2), padding=1, device=self.device),
            nn.BatchNorm3d(oc, device=self.device),
            # PW
            nn.Conv3d(oc, oc, (1, 1, 1), (1, 1, 1), device=self.device),
            nn.BatchNorm3d(oc, device=self.device),
            nn.LeakyReLU()
        )
        self.branch2 = nn.Sequential(
            # PW
            nn.Conv3d(in_channel, in_channel, (1, 1, 1), (1, 1, 1), device=self.device),
            nn.BatchNorm3d(in_channel, device=self.device),
            nn.LeakyReLU(),
            # DW
            nn.Conv3d(in_channel, oc, (3, 3, 3), (2, 2, 2), padding=1, device=self.device),
            nn.BatchNorm3d(oc, device=self.device),
            # PW
            nn.Conv3d(oc, oc, (1, 1, 1), (1, 1, 1), device=self.device),
            nn.BatchNorm3d(oc, device=self.device),
            nn.LeakyReLU()
        )

    def forward(self, x):
        # feed branch 1
        b1 = self.branch1.forward(x)
        # feed branch 2
        b2 = self.branch2.forward(x)
        # concat, shuffle and return
        x = torch.cat((b1, b2), dim=1)
        N, C, H, W, D = x.shape
        g = SHUFFLE_G
        return x.view(N, g, C // g, H, W, D).permute(0, 2, 1, 3, 4, 5).reshape(N, C, H, W, D)


class ShuffleGroup(nn.Module):
    def __init__(self, basic_unit_n, in_dim, out_dim, device='cpu'):
        super(ShuffleGroup, self).__init__()
        self.device = device
        self.group = nn.Sequential(OrderedDict([
            ('DownSampleUnit0', DownSampleUnit(in_dim, out_dim, self.device))
        ]))
        for i in range(basic_unit_n):
            self.group.add_module('BasicUnit{}'.format(i), BasicUnit(out_dim, self.device))

    def forward(self, x):
        x = self.group.forward(x)
        return x


class GridRegBlock(nn.Module):
    def __init__(self, device='cpu'):
        super(GridRegBlock, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv3d(SHUFFLE_CHS[-1], REG_CH, (1, 1, 1), (1, 1, 1), device=self.device),
            nn.BatchNorm3d(REG_CH, device=self.device),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.model.forward(x)
        return x


class GridCNN(nn.Module):
    def __init__(self, device='cpu'):
        super(GridCNN, self).__init__()
        self.device = device
        self.model = nn.Sequential(OrderedDict([
            # atom info integration block
            ('ChannelFuse', nn.Conv3d(GRID_CHANNELS, MAX_LIGAND_R,
                                      (1, 1, 1), (1, 1, 1), device=self.device)),
            ('MaxPool', nn.MaxPool3d(3, 2, padding=1))
        ]))
        # Feature Extraction Block (Shuffle Groups)
        for i in range(len(SHUFFLE_CHS) - 1):
            mod = ShuffleGroup(SHUFFLE_UNITS[i], SHUFFLE_CHS[i], SHUFFLE_CHS[i + 1], self.device)
            self.model.add_module('ShuffleGroup{}'.format(i), mod)
        # Grid Regression Block to process output as 1x1024 vectors
        self.model.add_module('GridRegBlock', GridRegBlock(self.device))

    def forward(self, x):
        xs = self.model.forward(x)
        return xs


if __name__ == '__main__':
    grid_cnn = GridCNN(device='cuda')
    summary(grid_cnn, (14, 36, 36, 36), 64)
