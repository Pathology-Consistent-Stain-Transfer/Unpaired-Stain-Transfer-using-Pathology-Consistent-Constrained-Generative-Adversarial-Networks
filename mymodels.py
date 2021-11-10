import torch.nn as nn
import torch.nn.functional as F
from unet_utils import Up, Down
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_features, alt_leak=False, neg_slope=1e-2):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Pathology_block(nn.Module):
    def __init__(self, in_features, out_features, n_residual_blocks, alt_leak=False, neg_slope=1e-2):
        super(Pathology_block, self).__init__()

        ext_model = [nn.Conv2d(in_features, out_features, 1),
                       nn.InstanceNorm2d(out_features),
                       nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True)]
        ext_model += [nn.ReflectionPad2d(1),
                       nn.Conv2d(out_features, out_features, 4, stride=2),
                       nn.InstanceNorm2d(out_features),
                       nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True)]

        for _ in range(n_residual_blocks):
            ext_model += [ResidualBlock(out_features, alt_leak, neg_slope)]
        self.extractor = nn.Sequential(*ext_model)

    def forward(self, x1, x2, x3):

        x1 = F.interpolate(x1, scale_factor=0.5)
        diffY1 = x2.size()[2] - x1.size()[2]
        diffX1 = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX1 // 2, diffX1 - diffX1 // 2,
                        diffY1 // 2, diffY1 - diffY1 // 2])
        x = torch.cat([x1, x2], dim=1)

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        diffY2 = x2.size()[2] - x3.size()[2]
        diffX2 = x2.size()[3] - x3.size()[3]
        x3 = F.pad(x3, [diffX2 // 2, diffX2 - diffX2 // 2,
                        diffY2 // 2, diffY2 - diffY2 // 2])
        x = torch.cat([x, x3], dim=1)

        return self.extractor(x)


class Generator_resnet(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=10, alt_leak=False, neg_slope=1e-2):
        super(Generator_resnet, self).__init__()

        # Initial convolution block [N 32 H W]
        model_encoder = [nn.ReflectionPad2d(3),
                         nn.Conv2d(input_nc, 32, 7),
                         nn.InstanceNorm2d(32),
                         nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True)]

        # Downsampling [N 64 H/2 W/2]-->[N 256 H/8 W/8]
        in_features = 32
        out_features = in_features*2
        for _ in range(3):
            model_encoder += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                              nn.InstanceNorm2d(out_features),
                              nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks [N 256 H/8 W/8]
        for _ in range(n_residual_blocks//2):
            model_encoder += [ResidualBlock(in_features, alt_leak, neg_slope)]

        model_decoder = []
        # Residual blocks [N 256 H/8 W/8]
        for _ in range(n_residual_blocks//2):
            model_decoder += [ResidualBlock(in_features, alt_leak, neg_slope)]
        # Upsampling [N 128 H/4 W/4]-->[N 32 H W]
        out_features = in_features//2
        for _ in range(3):
            model_decoder += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                              nn.InstanceNorm2d(out_features),
                              nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # Output layer [N 3 H W]
        model_decoder += [nn.ReflectionPad2d(3),
                          nn.Conv2d(32, output_nc, 7),
                          nn.Tanh()]

        self.encoder = nn.Sequential(*model_encoder)
        self.decoder = nn.Sequential(*model_decoder)

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output, features


class Generator_unet(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=8, alt_leak=False, neg_slope=1e-2):
        super(Generator_unet, self).__init__()
        # Initial convolution block [N 32 H W]
        self.inc = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(input_nc, 32, 7),
                                 nn.InstanceNorm2d(32),
                                 nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        # Downsampling [N 64 H/2 W/2]
        self.down1 = Down(32, 64, alt_leak, neg_slope)
        # Downsampling [N 128 H/4 W/4]
        self.down2 = Down(64, 128, alt_leak, neg_slope)
        # Downsampling [N 256 H/8 W/8]
        self.down3 = Down(128, 256, alt_leak, neg_slope)

        # Residual blocks [N 256 H/8 W/8]
        res_ext_encoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_encoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.ext_f1 = nn.Sequential(*res_ext_encoder)
        # Residual blocks [N 256 H/8 W/8]
        res_ext_decoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_decoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.ext_f2 = nn.Sequential(*res_ext_decoder)
        # Downsampling [N 128 H/4 W/4]
        self.up1 = Up(256, 128, alt_leak, neg_slope)
        # Downsampling [N 64 H/2 W/2]
        self.up2 = Up(128, 64, alt_leak, neg_slope)
        # Downsampling [N 32 H W]
        self.up3 = Up(64, 32, alt_leak, neg_slope)
        # Downsampling [N 3 H W]
        self.outc = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(32, output_nc, 7),
                                  nn.Tanh())

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        latent_features = self.ext_f1(x3)
        features = self.ext_f2(latent_features)
        x = self.up1(features, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)
        outputs = self.outc(x)
        return outputs, latent_features


class Generator_unet_cls(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=8, alt_leak=False, neg_slope=1e-2):
        super(Generator_unet_cls, self).__init__()
        # Initial convolution block [N 32 H W]
        self.inc = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(input_nc, 32, 7),
                                 nn.InstanceNorm2d(32),
                                 nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        # Downsampling [N 64 H/2 W/2]
        self.down1 = Down(32, 64, alt_leak, neg_slope)
        # Downsampling [N 128 H/4 W/4]
        self.down2 = Down(64, 128, alt_leak, neg_slope)
        # Downsampling [N 256 H/8 W/8]
        self.down3 = Down(128, 256, alt_leak, neg_slope)

        # Residual blocks [N 256 H/8 W/8]
        res_ext_encoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_encoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.ext_f1 = nn.Sequential(*res_ext_encoder)
        # merge features [N 256 H/8 W/8]
        self.pathology_f = Pathology_block(448, 256, n_residual_blocks // 2, alt_leak, neg_slope)

        self.merge = nn.Sequential(nn.Conv2d(512, 256, 1),
                                   nn.InstanceNorm2d(256),
                                   nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        # Residual blocks [N 256 H/8 W/8]
        res_ext_decoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_decoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.ext_f2 = nn.Sequential(*res_ext_decoder)
        # Downsampling [N 128 H/4 W/4]
        self.up1 = Up(256, 128, alt_leak, neg_slope)
        # Downsampling [N 64 H/2 W/2]
        self.up2 = Up(128, 64, alt_leak, neg_slope)
        # Downsampling [N 32 H W]
        self.up3 = Up(64, 32, alt_leak, neg_slope)
        # Downsampling [N 3 H W]
        self.outc = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(32, output_nc, 7),
                                  nn.Tanh())

        self.out_cls = nn.Sequential(nn.Dropout(0.15),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(256, 1, 3),
                                     nn.Sigmoid())

    def forward(self, x, mode='G'):
        # encoder
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # extract feature
        pathology_features = self.pathology_f(x1, x2, x3)
        c_out = self.out_cls(pathology_features)
        # Average pooling and flatten
        c_out = F.avg_pool2d(c_out, c_out.size()[2:]).view(c_out.size()[0])
        if mode == 'C':
            return c_out
        latent_features = self.ext_f1(x3)
        features = torch.cat([latent_features, pathology_features], dim=1)
        features = self.merge(features)
        features = self.ext_f2(features)

        # decoder
        x = self.up1(features, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)
        outputs = self.outc(x)
        return outputs, latent_features, c_out, pathology_features


class Generator_unet_seg(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=8, alt_leak=False, neg_slope=1e-2):
        super(Generator_unet_seg, self).__init__()
        # Initial convolution block [N 32 H W]
        self.inc = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(input_nc, 32, 7),
                                 nn.InstanceNorm2d(32),
                                 nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        # Downsampling [N 64 H/2 W/2]
        self.down1 = Down(32, 64, alt_leak, neg_slope)
        # Downsampling [N 128 H/4 W/4]
        self.down2 = Down(64, 128, alt_leak, neg_slope)
        # Downsampling [N 256 H/8 W/8]
        self.down3 = Down(128, 256, alt_leak, neg_slope)

        # Residual blocks [N 256 H/8 W/8]
        res_ext_encoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_encoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.ext_f1 = nn.Sequential(*res_ext_encoder)
        # merge features [N 256 H/8 W/8]
        self.pathology_f = Pathology_block(448, 256, n_residual_blocks // 2, alt_leak, neg_slope)

        self.merge = nn.Sequential(nn.Conv2d(512, 256, 1),
                                   nn.InstanceNorm2d(256),
                                   nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        # Residual blocks [N 256 H/8 W/8]
        res_ext_decoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_decoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.ext_f2 = nn.Sequential(*res_ext_decoder)
        # Downsampling [N 128 H/4 W/4]
        self.up1 = Up(256, 128, alt_leak, neg_slope)
        # Downsampling [N 64 H/2 W/2]
        self.up2 = Up(128, 64, alt_leak, neg_slope)
        # Downsampling [N 32 H W]
        self.up3 = Up(64, 32, alt_leak, neg_slope)
        # Downsampling [N 3 H W]
        self.outc = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(32, output_nc, 7),
                                  nn.Tanh())

        self.out_seg = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(256, 1, 3),
                                     nn.Sigmoid())

    def forward(self, x, mode='G'):
        # encoder
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # extract feature
        pathology_features = self.pathology_f(x1, x2, x3)
        c_out = self.out_seg(pathology_features)
        # Average pooling and flatten
        if mode == 'C':
            return c_out
        latent_features = self.ext_f1(x3)
        features = torch.cat([latent_features, pathology_features], dim=1)
        features = self.merge(features)
        features = self.ext_f2(features)

        # decoder
        x = self.up1(features, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)
        outputs = self.outc(x)
        return outputs, latent_features, c_out, pathology_features


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1 ),
                    nn.LeakyReLU(0.2, inplace=True)]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])