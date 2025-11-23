import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== DoubleConv =====
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
    def forward(self, x):
        return self.double_conv(x)

# ===== Down =====
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout)
        )
    def forward(self, x):
        return self.pool_conv(x)

# ===== Up =====
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                            diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ===== UNet6 =====
class UNet6(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=16, dropout=0.0):
        super().__init__()
        ch = base_ch
        self.inc = DoubleConv(in_channels, ch, dropout=dropout)      # 6x6
        self.down1 = Down(ch, ch*2, dropout=dropout)                # 3x3
        self.bottleneck = DoubleConv(ch*2, ch*4, dropout=dropout)   # 3x3
        self.up1 = Up(ch*4, ch*2, skip_ch=ch, dropout=dropout)      # 6x6
        self.final_conv = nn.Conv2d(ch*2, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.bottleneck(x2)
        x = self.up1(x3, x1)
        out = self.final_conv(x)
        return out
