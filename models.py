import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        # concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DualBranchUNet(nn.Module):
    """
    Dual-Branch Generator:
    - Shares a common Encoder.
    - Decodes to Enhancement output (1 channel).
    - Decodes to Colorization output (3 channels), fusing features from the Enhancement decoder.
    """
    def __init__(self, n_channels=1, bilinear=False):
        super(DualBranchUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        # Shared Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # -------------------------------------------------------------
        # Enhancement Decoder (Grayscale Reconstruction)
        # -------------------------------------------------------------
        self.enh_up1 = Up(1024, 512 // factor, bilinear)
        self.enh_up2 = Up(512, 256 // factor, bilinear)
        self.enh_up3 = Up(256, 128 // factor, bilinear)
        self.enh_up4 = Up(128, 64, bilinear)
        self.enh_outc = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Tanh() # normalized range [-1, 1]
        )

        # -------------------------------------------------------------
        # Colorization Decoder (RGB Reconstruction with Feature Fusion)
        # -------------------------------------------------------------
        # For the colorization decoder, at each scale, we concatenate:
        # 1. The upsampled feature map.
        # 2. The encoder skip connection.
        # 3. The corresponding feature map from the enhancement branch decoder (Feature Fusion!)
        # Thus, in_channels for colorization decoder blocks:
        # color_up1 takes: upsampled (512) + encoder skip (512) + enhancement features (512) = 1536 channels -> outputs 256
        # To keep it standard and clean, we will concatenate the features.
        self.col_up1 = Up(1024 + 512, 512 // factor, bilinear) # taking input from bottleneck (1024), enc3 (512), and enh_up1 output (512)
        self.col_up2 = Up(512 + 256, 256 // factor, bilinear)  # taking input from col_up1 (512), enc2 (256), and enh_up2 output (256)
        self.col_up3 = Up(256 + 128, 128 // factor, bilinear)  # taking input from col_up2 (256), enc1 (128), and enh_up3 (128)
        self.col_up4 = Up(128 + 64, 64, bilinear)               # taking input from col_up3 (128), inc (64), and enh_up4 (64)
        
        self.col_outc = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh() # normalized range [-1, 1]
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 1. Enhancement Branch
        # We need to save the intermediate decoder feature maps to pass to the colorization branch
        enh_f1 = self.enh_up1(x5, x4) # outputs 512 channels
        enh_f2 = self.enh_up2(enh_f1, x3) # outputs 256 channels
        enh_f3 = self.enh_up3(enh_f2, x2) # outputs 128 channels
        enh_f4 = self.enh_up4(enh_f3, x1) # outputs 64 channels
        enh_out = self.enh_outc(enh_f4)

        # 2. Colorization Branch with Feature Fusion
        # We perform feature fusion by concatenating the enhancement decoder features at each step
        
        # col_up1 input: bottleneck x5 upsampled, fused with x4 (encoder) and enh_f1 (enhancement decoder)
        # Note: Up module performs self.up(x1) and cats with x2.
        # Here we pass x5 as x1, and the concatenation of [x4, enh_f1] as x2.
        fused_x4 = torch.cat([x4, enh_f1], dim=1)
        col_f1 = self.col_up1(x5, fused_x4)
        
        fused_x3 = torch.cat([x3, enh_f2], dim=1)
        col_f2 = self.col_up2(col_f1, fused_x3)
        
        fused_x2 = torch.cat([x2, enh_f3], dim=1)
        col_f3 = self.col_up3(col_f2, fused_x2)
        
        fused_x1 = torch.cat([x1, enh_f4], dim=1)
        col_f4 = self.col_up4(col_f3, fused_x1)
        
        col_out = self.col_outc(col_f4)

        return enh_out, col_out

class PatchGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator (Pixel2Pixel style)"""
    def __init__(self, input_nc=1, target_nc=3, ndf=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        # input_nc is 1 (IR image), target_nc is 3 (RGB image)
        # Conditioned on the input image, so in_channels = input_nc + target_nc
        in_channels = input_nc + target_nc
        
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Output 1-channel classification map
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*layers)

    def forward(self, input_ir, target_rgb):
        # Concatenate condition (IR) and target (RGB) along channel dimension
        x = torch.cat([input_ir, target_rgb], dim=1)
        return self.model(x)

if __name__ == "__main__":
    # Test generator and discriminator dimensions
    ir_in = torch.randn(2, 1, 256, 256)
    rgb_in = torch.randn(2, 3, 256, 256)
    
    netG = DualBranchUNet(n_channels=1)
    netD = PatchGANDiscriminator(input_nc=1, target_nc=3)
    
    enh, col = netG(ir_in)
    pred = netD(ir_in, col)
    
    print(f"Generator - Enh output shape: {enh.shape}, Col output shape: {col.shape}")
    print(f"Discriminator - output shape: {pred.shape}")
