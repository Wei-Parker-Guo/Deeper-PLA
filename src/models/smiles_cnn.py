# this module constructs the entire pipeline for forwarding smiles
from collections import OrderedDict

import torch
from torch import nn
from torchsummary import summary

from src.global_vars import SMILES_CNN_G, SMILES_CNN_CH, REG_CH


# a simple residual block to replace traditional CNN to avoid vanishing gradients
# this residual block follows the pre-activation format proposed by K. He in 2016
class ResidualBlock(nn.Module):
    def __init__(self, dim, ksize, stride, pad, device='cpu'):
        super(ResidualBlock, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.BatchNorm2d(dim, device=self.device),
            nn.ReLU(),
            nn.Conv2d(dim, dim, ksize, stride, pad, device=self.device),
            nn.BatchNorm2d(dim, device=self.device),
            nn.ReLU(),
            nn.Conv2d(dim, dim, ksize, stride, pad, device=self.device)
        )

    def forward(self, x):
        x2 = self.model.forward(x)
        x = torch.add(x, x2)
        return x


# residual block with down sampling
class DSResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ksize1, ksize2, stride1, stride2, pad, device='cpu'):
        super(DSResidualBlock, self).__init__()
        self.device = device
        self.b1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, (1, 1), (1, 1), device=self.device),
            nn.AvgPool2d(ksize1, stride1, pad)
        )
        self.b2 = nn.Sequential(
            nn.BatchNorm2d(in_dim, device=self.device),
            nn.ReLU(),
            nn.Conv2d(in_dim, out_dim, ksize1, stride1, pad, device=self.device),
            nn.BatchNorm2d(out_dim, device=self.device),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, ksize2, stride2, pad, device=self.device)
        )

    def forward(self, x):
        x1 = self.b1.forward(x)
        x2 = self.b2.forward(x)
        return torch.add(x1, x2)


class ResidualGroup(nn.Module):
    def __init__(self, in_dim, out_dim, block_n=3, device='cpu'):
        super(ResidualGroup, self).__init__()
        self.device = device
        self.model = nn.Sequential(OrderedDict([
            ('DSResBlock', DSResidualBlock(in_dim, out_dim,
                                           (3, 3), (3, 3), (2, 2), (1, 1), 1, self.device))
        ]))
        for i in range(block_n):
            self.model.add_module('ResBlock{}'.format(i),
                                  ResidualBlock(out_dim, (3, 3), (1, 1), 1, self.device))

    def forward(self, x):
        return self.model.forward(x)


class SmilesCNN(nn.Module):
    def __init__(self, device='cpu'):
        super(SmilesCNN, self).__init__()
        self.device = device
        self.model = nn.Sequential(OrderedDict([
            ('GlobalConv', nn.Conv2d(SMILES_CNN_CH[0], SMILES_CNN_CH[1], (7, 7), (2, 2), 1, device=self.device)),
            ('Pool', nn.AvgPool2d((3, 3), (2, 2), padding=1))
        ]))
        for i in range(1, len(SMILES_CNN_CH) - 1):
            g = ResidualGroup(SMILES_CNN_CH[i], SMILES_CNN_CH[i+1], SMILES_CNN_G[i-1], self.device)
            self.model.add_module('ResGroup{}'.format(i-1), g)
        # head section
        self.model.add_module('Dropout', nn.Dropout(0.2))
        self.model.add_module('HeadConv',
                              nn.Conv2d(SMILES_CNN_CH[-1], REG_CH, (3, 3), (1, 1),
                                        padding=1, device=self.device))
        self.model.add_module('HeadAvgPool', nn.AvgPool2d((3, 3), (1, 1), padding=1))

    def forward(self, x):
        return self.model.forward(x)


if __name__ == '__main__':
    model = SmilesCNN(device='cuda')
    summary(model, (1, 76, 76), 64)
