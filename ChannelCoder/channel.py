import torch.nn as nn
import numpy as np
import os
import torch


class Channel(nn.Module):
    def __init__(self, args):
        super(Channel, self).__init__()
        self.args = args
        self.chan_type = args.chan_type
        self.chan_param = args.chan_param  # SNR
        # if args.logger:
        #     args.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
        #         args.channel['type'], args.channel['chan_param']))

    def gaussian_noise_layer(self, input_layer, std):
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        noise = noise.to(input_layer.get_device())
        return input_layer + noise

    def complex_normalize(self, x, power):    #使功率变为1
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr

    def complex_forward(self, channel_in):
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            # power normalization
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma)
            return chan_output

    def forward(self, input, H, W):
    # def forward(self, input):
        # input \in R
        channel_tx, pwr = self.complex_normalize(input, power=1)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]                                     #modulation
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j
        channel_output = self.complex_forward(channel_in)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)        #demodulation

        noise = (channel_output - channel_tx).detach()
        noise.requires_grad = False
        channel_rx = channel_tx + noise
        return channel_rx, H, W
        # return channel_rx
