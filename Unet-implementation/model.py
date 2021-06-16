import torch
import torch.nn as nn
from math import sqrt
from torchsummary import summary
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        DoubleConv.in_channels = in_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_block.apply(DoubleConv.weights_init)

    @staticmethod
    def weights_init(layer):
        if type(layer) == nn.Conv2d:
            torch.nn.init.normal_(
                layer.weight, mean=0.0, std=sqrt(2 / 9 * DoubleConv.in_channels)
            )

    def forward(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.3, inplace=True)

        self.encoder_1 = DoubleConv(in_channels=in_channels, out_channels=64)
        self.encoder_2 = DoubleConv(in_channels=64, out_channels=128)
        self.encoder_3 = DoubleConv(in_channels=128, out_channels=256)
        self.encoder_4 = DoubleConv(in_channels=256, out_channels=512)

        self.bottle_neck = DoubleConv(in_channels=512, out_channels=1024)

    def forward(self, x):
        f1 = self.encoder_1(x)
        p1 = self.max_pooling(f1)
        p1 = self.dropout(p1)

        f2 = self.encoder_2(p1)
        p2 = self.max_pooling(f2)
        p2 = self.dropout(p2)

        f3 = self.encoder_3(p2)
        p3 = self.max_pooling(f3)
        p3 = self.dropout(p3)

        f4 = self.encoder_4(p3)
        p4 = self.max_pooling(f4)
        p4 = self.dropout(p4)

        p5 = self.bottle_neck(p4)
        return p5, (f1, f2, f3, f4)


class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
        self.dropout = nn.Dropout2d(p=0.3, inplace=True)

    def forward(self, x, conv_output):

        x = self.conv_transpose(x)

        if x.shape != conv_output.shape:
            x = TF.resize(x, size=conv_output.shape[2:])

        x = torch.cat((conv_output, x), dim=1)
        x = self.dropout(x)
        x = self.conv(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_4 = Decoder_Block(in_channels=1024, out_channels=512)
        self.decoder_3 = Decoder_Block(in_channels=512, out_channels=256)
        self.decoder_2 = Decoder_Block(in_channels=256, out_channels=128)
        self.decoder_1 = Decoder_Block(in_channels=128, out_channels=64)

    def forward(self, inputs, conv_inputs):
        f1, f2, f3, f4 = conv_inputs

        x = self.decoder_4(inputs, f4)
        x = self.decoder_3(x, f3)
        x = self.decoder_2(x, f2)
        x = self.decoder_1(x, f1)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder()
        self.final_layer = nn.Conv2d(
            in_channels=64, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        p, intermediates = self.encoder(x)
        out = self.decoder(p, intermediates)
        out = self.final_layer(out)
        return out


if __name__ == "__main__":
    dummy = torch.zeros(1, 3, 256, 256)
    layer = UNet(in_channels=3, out_channels=1)
    summary(layer, dummy)

