import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


# Define the basic building blocks: Encoder Block,Bottleneck Block, Decoder Block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.encode(x)
        pool = self.pool(x)
        return pool, x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckBlock, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bottleneck(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, bridge):
        x = self.upconv(x)
        x = torch.cat((x, bridge), dim=1)
        x = self.decode(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=7):
        super(UNet, self).__init__()
        # Encoder
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)

        # Bottleneck
        self.bottleneck = BottleneckBlock(256, 512)

        # Decoder
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)

        # Output layer
        self.outconv = nn.Conv2d(64, 16, kernel_size=1)
        self.avgpool = nn.AvgPool2d(5, stride=2)
        self.fc = nn.Linear(7744, out_channels)

    def forward(self, x):
        # Encoder
        x, pool1 = self.encoder1(x)
        x, pool2 = self.encoder2(x)
        x, pool3 = self.encoder3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder3(x, pool3)
        x = self.decoder2(x, pool2)
        x = self.decoder1(x, pool1)

        # Output
        x = self.outconv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = F.log_softmax(x, dim=-1)
        return x