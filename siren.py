import torch
from torch import nn
from math import sqrt
import torch.nn.functional as F
from ChannelCoder.DeepJSCC import DeepJSCC


class FeatureGrid(nn.Module):
    def __init__(self, fdim, fsize, ftype):
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        self.ftype = ftype
        if self.ftype == '2D':
            self.fm = nn.Parameter(torch.randn(1, fdim, fsize, fsize))
        elif self.ftype == '3D':
            self.fm = nn.Parameter(torch.randn(1, fdim, fsize+1, fsize+1, fsize+1) * 0.01)
        else:
            self.fm = nn.Parameter(torch.randn(1, fdim, fsize))
        self.sparse = None

    def forward(self, x_coords):
        # N = x_coords.shape[0]
        if self.ftype== '2D':
            sample = F.grid_sample(self.fm, x_coords,
                                   align_corners=True, padding_mode='border')[0,:,:,:].permute(1, 2, 0)
        elif self.ftype == '3D':
            sample = F.grid_sample(self.fm, x_coords, 
                                   align_corners=True, padding_mode='border')[0,:,:,:,:].transpose(0, -1)
        else:
            sample = F.grid_sample(self.fm, x_coords,
                                   align_corners=True, padding_mode='border')[0, :, :].transpose(0, -1)
        return sample

class SynthesisTransform(nn.Module):
    def __init__(self, input_dim, dim_hidden, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, dim_hidden), nn.ReLU()]
        if num_layers > 2:
            for i in range(num_layers - 2):
                layers += [nn.Linear(dim_hidden, dim_hidden), nn.ReLU()]
        layers += [nn.Linear(dim_hidden, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_dim = 2
        self.fdim = self.args.feature_dim
        self.hidden_dim = self.args.layer_size
        self.num_layers = self.args.num_layers
        self.features = nn.ModuleList()
        for i in range(self.args.num_lods):
            self.features.append(FeatureGrid(self.fdim, (2**(i+self.args.base_lod)), args.input_type))
        # self.log_gains = nn.Parameter(
        #     torch.arange(1., self.args.num_lods+1), requires_grad=True
        # )
        if self.args.sample_type == 'cat':
            self.siren_input_dim = self.args.num_lods * self.fdim
        if self.args.sample_type == 'sum':
            self.siren_input_dim = self.fdim + self.input_dim
        if self.args.input_type == '2D':
            self.dim_out = 3
        else:
            self.dim_out = 1
        self.net = SynthesisTransform(self.siren_input_dim, self.hidden_dim, self.num_layers)
        self.deepJSCC = DeepJSCC(args, self.fdim)

    def pass_channel(self):
        distortions = []
        for i in range(self.args.num_lods):
            single_feature = self.features[i].fm.data
            single_feature_hat, distortion_single = self.deepJSCC(single_feature)
            self.features[i].fm.data = single_feature_hat
            distortions.append(distortion_single)
        return torch.stack(distortions, dim=0).mean()

    def encode(self, x_coords, return_lst):
        samples = []
        l = []
        for i in range(self.args.num_lods):
            self.features[i].fm.data = self.features[i].fm.data * torch.pow(2, self.log_gains[i])
            # Query features
            # self.features[i].fm.data = self.deepJSCC(self.features[i].fm.data)
            sample = self.features[i](x_coords)
            samples.append(sample)

            # Sum queried features
            if i > 0:
                if self.args.sample_type == 'cat':
                    samples[i] = torch.cat((samples[i-1], samples[i]), dim=-1)
                elif self.args.sample_type == 'sum':
                    samples[i] += samples[i-1]

            # Concatenate coordinates
            # ex_sample = samples[i]
            # x_coords_squeeze = x_coords.squeeze()
            # ex_sample = torch.cat([x_coords_squeeze, ex_sample], dim=-1)
            #
            # values = self.net(ex_sample)
            # l.append(values)
        ex_sample = samples[-1]
        # x_coords_squeeze = x_coords.squeeze()
        # ex_sample = torch.cat([x_coords_squeeze, ex_sample], dim=-1)

        values = self.net(ex_sample)
        l.append(values)
        if return_lst:
            return l
        else:
            return l[-1]