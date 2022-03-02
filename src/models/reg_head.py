# The regression head that takes in both smiles and grid cnn outputs.
# This is essentially just a shared weight FC layer.
import torch
from torch import nn
from src.global_vars import REG_CH


class RegHead(nn.Module):
    def __init__(self, device='cpu'):
        super(RegHead, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(REG_CH, 1, device=self.device),
            nn.Sigmoid()
        )

    def forward(self, x):
        # slice x into vectors with size of (1024,), repetitively forward with shared weights
        bs = x.shape[0]
        if self.training:
            xs = x.reshape(-1, REG_CH)
            r = self.model.forward(xs)
            return r.reshape(bs, -1)
        # if not training, average vectors into one and do prediction
        else:
            xs = x.reshape(bs, -1, REG_CH)
            xs = torch.mean(xs, dim=1)
            r = self.model.forward(xs)
            return r


if __name__ == '__main__':
    model = RegHead()
    model.train(True)
    x = torch.randn((4, 1024 * (3 * 3 * 3 + 5 * 5)))
    x = model.forward(x)
    print(x)
