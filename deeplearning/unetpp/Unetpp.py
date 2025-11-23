import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Double Conv with BatchNorm"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetPP6(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16):
        super().__init__()
        ch = base_ch
        # Encoder
        self.x0_0 = ConvBlock(in_ch, ch)
        self.x1_0 = ConvBlock(ch, ch*2)
        self.x2_0 = ConvBlock(ch*2, ch*4)
        # MaxPool
        self.pool = nn.MaxPool2d(2)
        # Nested skip conv
        self.x0_1 = ConvBlock(ch + ch*2, ch)
        self.x1_1 = ConvBlock(ch*2 + ch*4, ch*2)
        self.x0_2 = ConvBlock(ch + ch, ch)
        # Up Convs
        self.up1_0 = nn.ConvTranspose2d(ch*2, ch, 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(ch*2, ch, 2, stride=2)
        # Final
        self.final_conv = nn.Conv2d(ch, out_ch, 1)

    def forward(self, x):
        # Encoder
        x0_0 = self.x0_0(x)       # 6x6
        x1_0 = self.x1_0(self.pool(x0_0))  # 3x3
        x2_0 = self.x2_0(self.pool(x1_0))  # 1x1 or 3x3, bottleneck 유지

        # Decoder Nested Skip
        x1_0_up = self.up2_0(x2_0)
        x1_1 = self.x1_1(torch.cat([x1_0, x1_0_up], dim=1))
        
        x0_0_up = self.up1_0(x1_1)
        x0_1 = self.x0_1(torch.cat([x0_0, x0_0_up], dim=1))
        
        x0_1_up = self.up1_1(x0_1)
        x0_2 = self.x0_2(torch.cat([x0_0, x0_1_up], dim=1))
        
        out = self.final_conv(x0_2)
        return out
