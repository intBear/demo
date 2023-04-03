import torch
import torch.nn as nn
from normalization.GDN import GDN


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, tcn):
        super(FeatureEncoder, self).__init__()
        self.input_dim = input_dim
        self.tcn = tcn
        device = torch.device('cuda')

        # (C, H, W) -> (256, H/2, W/2)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=(5, 5), stride=2, padding=(5 - 2) // 2+1),
            # GDN(256, device, False),
            nn.PReLU()
        )
        # (256, H/2, W/2) -> (256, H/4, W/4)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=(3 - 2) // 2+1),
            # GDN(256, device, False),
            nn.PReLU()
        )
        # (256, H/4, W/4) -> (256, H/4, W/4)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=(3 - 1) // 2),
            # GDN(256, device, False),
            nn.PReLU()
        )
        # (256, H/4, W/4) -> (tcn, H/4, W/4)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, tcn, kernel_size=(3, 3), stride=1, padding=(3 - 1) // 2),
            # GDN(tcn, device, False),
        )

    def forward(self, x):
        H = int(x.shape[2] / 4)
        W = int(x.shape[3] / 4)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        out = x.reshape(-1, self.tcn * H * W)
        return out, H, W