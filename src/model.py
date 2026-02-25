# src/model.py
import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.2),  
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)

        # Bottleneck
        self.bottleneck = conv_block(128, 256)

        # Decoder
        self.up3    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3   = conv_block(256, 128)

        self.up2    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2   = conv_block(128, 64)

        self.up1    = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1   = conv_block(64, 32)

        # Output
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


if __name__ == '__main__':
    model = UNet(in_channels=4, out_channels=1)
    x = torch.randn(2, 4, 64, 64)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
