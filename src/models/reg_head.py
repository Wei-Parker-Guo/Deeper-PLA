# The regression head that takes in both smiles and grid cnn outputs.
# This is essentially just a shared weight FC layer.
import torch
from torch import nn
from src.global_vars import REG_CH


class RegHead(nn.Module):
    def __init__(self, device='cpu'):
        super(RegHead, self).__init__()
        self.device = device
        self.model = nn.Linear(REG_CH, 1, device=self.device)

    def forward(self, x):
        # slice x into vectors with size of (1024,), repetitively forward with shared weights
        xs = torch.split(x, REG_CH, dim=1)
        r = torch.Tensor([])
        for i in xs:
            r = torch.cat((r, self.model.forward(i)), dim=1)
        return r


if __name__ == '__main__':
    model = RegHead()
    x = torch.zeros((4, 1024 * (3 * 3 * 3 + 5 * 5)))
    x = model.forward(x)
    print(x.size())
