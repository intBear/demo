from ChannelCoder.ChannelDecoder import *
from ChannelCoder.ChannelEncoder import *
from ChannelCoder.channel import Channel


class DeepJSCC(nn.Module):
    def __init__(self, args, input_dim):
        super(DeepJSCC, self).__init__()
        # if args.logger:
        #     args.logger.info("【Network】: Built Distributed JSCC model, C={}, k/n={}".format(args.tcn, args.kdivn))

        self.Encoder = FeatureEncoder(input_dim, args.tcn)
        self.Decoder = FeatureDecoder(input_dim, args.tcn)
        self.channel = Channel(args)
        self.pass_channel = args.pass_channel
        self.distortion_loss = torch.nn.MSELoss()

    def feature_pass_channel(self, feature, H, W):
        noisy_feature, H, W = self.channel(feature, H, W)
        return noisy_feature, H, W

    def forward(self, input_image):
        feature, H, W = self.Encoder(input_image)
        if self.pass_channel:
            noisy_feature, H, W = self.feature_pass_channel(feature, H, W)
        else:
            noisy_feature = feature
        recon_image = self.Decoder(noisy_feature, H, W)
        # recon_image = self.Decoder(noisy_feature)

        distortion_loss = self.distortion_loss.forward(input_image, recon_image)
        return recon_image, distortion_loss