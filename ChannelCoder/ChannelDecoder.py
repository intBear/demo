import torch
import torch.nn as nn
from normalization.GDN import GDN


class FeatureDecoder(nn.Module):
    def __init__(self, input_dim, tcn, activation='prelu', channel_norm=True):
        super(FeatureDecoder, self).__init__()
        self.tcn = tcn
        self.input_dim = input_dim
        device = torch.device('cuda')
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU', prelu='PReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu, prelu)
        self.sigmoid = nn.Sigmoid()

        # (C, H/4, W/4) -> (256, H/4, W/4)
        self.upconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(tcn, 256, kernel_size=(3, 3), stride=1, padding=(3 - 1) // 2),
            # GDN(256, device, True),
            self.activation(),
        )

        # (256, H/4, W/4) -> (256, H/4, W/4)
        self.upconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=1, padding=(3 - 1) // 2),
            # GDN(256, device, True),
            self.activation(),
        )

        # (256, H/4, W/4) -> (256, H/2, W/2)
        self.upconv_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=2, padding=(3 - 2) // 2 + 1, output_padding=1),
            # GDN(256, device, True),
            self.activation(),
        )

        # (256, H/4, W/4) -> (input_dim, H, W)
        self.upconv_block4 = nn.Sequential(
            nn.ConvTranspose2d(256, self.input_dim, kernel_size=(5, 5), stride=2, padding=(5 - 2) // 2 + 1, output_padding=1),
            # GDN(self.input_dim, device, True),
            nn.Sigmoid(),
        )

    def forward(self, x, H, W):
    # def forward(self, x):
        # x = x.reshape(-1, self.tcn, x1, x2)
        x = x.reshape(-1, self.tcn, H, W)
        x = self.upconv_block1(x)
        x = self.upconv_block2(x)
        x = self.upconv_block3(x)
        out = self.upconv_block4(x)
        return out