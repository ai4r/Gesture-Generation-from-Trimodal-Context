import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


""" reimplement speech2gesture model(https://github.com/amirbar/speech2gesture) with pytorch """

class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    from https://github.com/mlperf/inference/blob/482f6a3beb7af2fb0bd2d91d6185d5e71c22c55f/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv1d_tf(nn.Conv1d):
    """
    Conv1d with the padding behavior from TF
    modified from https://github.com/mlperf/inference/blob/482f6a3beb7af2fb0bd2d91d6185d5e71c22c55f/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py
    """

    def __init__(self, *args, **kwargs):
        super(Conv1d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        if rows_odd:
            input = F.pad(input, [0, rows_odd])

        return F.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


def ConvNormRelu(in_channels, out_channels, type='1d', downsample=False, k=None, s=None, padding='SAME'):
    if k is None and s is None:
        if not downsample:
            k = 3
            s = 1
        else:
            k = 4
            s = 2

    if type == '1d':
        conv_block = Conv1d_tf(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
        norm_block = nn.BatchNorm1d(out_channels)
    elif type == '2d':
        conv_block = Conv2d_tf(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
        norm_block = nn.BatchNorm2d(out_channels)
    else:
        assert False

    return nn.Sequential(
        conv_block,
        norm_block,
        nn.LeakyReLU(0.2, True)
    )


class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetUp, self).__init__()
        self.conv = ConvNormRelu(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = torch.repeat_interleave(x1, 2, dim=2)
        x1 = x1[:, :, :x2.shape[2]]  # to match dim
        x = x1 + x2  # it is different to the original UNET, but I stick to speech2gesture implementation
        x = self.conv(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_frames):
        super().__init__()
        self.n_frames = n_frames
        self.first_net = nn.Sequential(
            ConvNormRelu(1, 64, '2d', False),
            ConvNormRelu(64, 64, '2d', True),
            ConvNormRelu(64, 128, '2d', False),
            ConvNormRelu(128, 128, '2d', True),
            ConvNormRelu(128, 256, '2d', False),
            ConvNormRelu(256, 256, '2d', True),
            ConvNormRelu(256, 256, '2d', False),
            ConvNormRelu(256, 256, '2d', False, padding='VALID')
        )

        self.make_1d = torch.nn.Upsample((n_frames, 1), mode='bilinear', align_corners=False)

        self.down1 = nn.Sequential(
            ConvNormRelu(256, 256, '1d', False),
            ConvNormRelu(256, 256, '1d', False)
        )
        self.down2 = ConvNormRelu(256, 256, '1d', True)
        self.down3 = ConvNormRelu(256, 256, '1d', True)
        self.down4 = ConvNormRelu(256, 256, '1d', True)
        self.down5 = ConvNormRelu(256, 256, '1d', True)
        self.down6 = ConvNormRelu(256, 256, '1d', True)
        self.up1 = UnetUp(256, 256)
        self.up2 = UnetUp(256, 256)
        self.up3 = UnetUp(256, 256)
        self.up4 = UnetUp(256, 256)
        self.up5 = UnetUp(256, 256)

    def forward(self, spectrogram):
        spectrogram = spectrogram.unsqueeze(1)  # add channel dim
        # print(spectrogram.shape)
        spectrogram = spectrogram.float()

        out = self.first_net(spectrogram)
        out = self.make_1d(out)
        x1 = out.squeeze(3)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)

        return x


class Generator(nn.Module):
    def __init__(self, n_poses, pose_dim, n_pre_poses):
        super().__init__()
        self.gen_length = n_poses

        self.audio_encoder = AudioEncoder(n_poses)
        self.pre_pose_encoder = nn.Sequential(
            nn.Linear(n_pre_poses * pose_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            ConvNormRelu(256 + 16, 256),
            ConvNormRelu(256, 256),
            ConvNormRelu(256, 256),
            ConvNormRelu(256, 256)
        )
        self.final_out = nn.Conv1d(256, pose_dim, 1, 1)

    def forward(self, in_spec, pre_poses):
        audio_feat_seq = self.audio_encoder(in_spec)  # output (bs, feat_size, n_frames)
        pre_poses = pre_poses.reshape(pre_poses.shape[0], -1)
        pre_pose_feat = self.pre_pose_encoder(pre_poses)  # output (bs, 16)
        pre_pose_feat = pre_pose_feat.unsqueeze(2).repeat(1, 1, self.gen_length)
        feat = torch.cat((audio_feat_seq, pre_pose_feat), dim=1)
        out = self.decoder(feat)
        out = self.final_out(out)
        out = out.transpose(1, 2)  # to (batch, seq, dim)

        return out


class Discriminator(nn.Module):
    def __init__(self, pose_dim):
        super().__init__()
        self.net = nn.Sequential(
            Conv1d_tf(pose_dim, 64, kernel_size=4, stride=2, padding='SAME'),
            nn.LeakyReLU(0.2, True),
            ConvNormRelu(64, 128, '1d', True),
            ConvNormRelu(128, 256, '1d', k=4, s=1),
            Conv1d_tf(256, 1, kernel_size=4, stride=1, padding='SAME'),
        )

    def forward(self, x):
        x = x[:, 1:] - x[:, :-1]  # pose differences
        x = x.transpose(1, 2)  # to (batch, dim, seq)

        out = self.net(x)
        return out


if __name__ == '__main__':
    # for model debugging
    pose_dim = 16
    generator = Generator(64, pose_dim, 4)
    spec = torch.randn((4, 128, 64))
    pre_poses = torch.randn((4, 4, pose_dim))
    generated = generator(spec, pre_poses)
    print('spectrogram', spec.shape)
    print('output', generated.shape)

    discriminator = Discriminator(pose_dim)
    out = discriminator(generated)
    print('discrimination output', out.shape)

