""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):
    """Downscaling with conv with stride=2,instanceNorm, relu"""

    def __init__(self, in_features, out_features, alt_leak=False, neg_slope=1e-2):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_features, out_features, alt_leak=False, neg_slope=1e-2):
        super().__init__()
        # upsample
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.gate = nn.Sequential(
            nn.Conv2d(out_features, out_features//2, 1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))

        self.merge = nn.Sequential(
            nn.Conv2d(out_features+out_features//2, out_features, 1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))

    def forward(self, x1, x2):

        x1 = self.up_conv(x1)

        x2 = self.gate(x2)
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)

        return self.merge(x)

class Pathology_feature(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_features, out_features, alt_leak=False, neg_slope=1e-2):
        super().__init__()
        # upsample
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.merge = nn.Sequential(
            nn.Conv2d(in_features, out_features, 1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3):

        x1 = self.up4(x1)

        diffY1 = x3.size()[2] - x1.size()[2]
        diffX1 = x3.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX1 // 2, diffX1 - diffX1 // 2,
                        diffY1 // 2, diffY1 - diffY1 // 2])

        x = torch.cat([x3, x1], dim=1)

        x2 = self.up2(x2)

        diffY2 = x3.size()[2] - x2.size()[2]
        diffX2 = x3.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX2 // 2, diffX2 - diffX2 // 2,
                        diffY2 // 2, diffY2 - diffY2 // 2])

        x = torch.cat([x, x2], dim=1)

        return self.merge(x)

