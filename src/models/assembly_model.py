# the assembled model from input to prediction
import torch
from torch import nn

from src.models.smiles_cnn import SmilesCNN
from src.models.grid_cnn import GridCNN
from src.models.reg_head import RegHead


class AssemblyModel(nn.Module):
    def __init__(self, device='cpu'):
        super(AssemblyModel, self).__init__()
        self.device = device
        self.smiles_cnn = SmilesCNN(self.device)
        self.grid_cnn = GridCNN(self.device)
        self.reg_head = RegHead(self.device)

    # x1 is grid and x2 is smiles
    def forward(self, x1, x2):
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        g = torch.flatten(self.grid_cnn.forward(x1), start_dim=1)
        s = torch.flatten(self.smiles_cnn.forward(x2), start_dim=1)
        x = torch.cat((g, s), dim=1)
        x = self.reg_head.forward(x)
        return x


if __name__ == '__main__':
    model = AssemblyModel()
    xt1 = torch.zeros((4, 14, 36, 36, 36))
    xt2 = torch.zeros((4, 1, 76, 76))
    xt = model.forward(xt1, xt2)
    print(xt.size())
