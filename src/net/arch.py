import torch
import torch.nn as nn
import numpy as np

L_CENTER = 50.0
L_RANGE = 100.0
AB_RANGE = 110.0

def normalize_l(l):
    return (l - L_CENTER) / L_RANGE

class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()

        self.conv1 = nn.Sequential(
            *[
                nn.Conv2d(1, 64, 3, padding=1, stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1, stride=2, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
            ]
        )

        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(64, 128, 3, padding=1, stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1, stride=2, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
            ]
        )

        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(128, 256, 3, padding=1, stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1, stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1, stride=2, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
            ]
        )

        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(256, 512, 3, padding=1, stride=1, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1, stride=1, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1, stride=1, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(512),
            ]
        )

        self.conv5 = nn.Sequential(
            *[
                nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(512),
            ]
        )

        self.conv6 = nn.Sequential(
            *[
                nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=2, stride=1, dilation=2, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(512),
            ]
        )

        self.conv7 = nn.Sequential(
            *[
                nn.Conv2d(512, 512, 3, padding=1, stride=1, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1, stride=1, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1, stride=1, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(512),
            ]
        )

        self.conv8 = nn.Sequential(
            *[
                nn.ConvTranspose2d(512, 256, 4, padding=1, stride=2, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1, stride=1, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1, stride=1, dilation=1, bias=True),
                nn.ReLU(inplace=True),
            ]
        )

        self.softmax = nn.Sequential(
            *[
                nn.Conv2d(256, 326, 1, padding=0, stride=1, dilation=1, bias=True),
                # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                # nn.Softmax(dim=1),
            ]
        )

    def forward(self, x):
        x = normalize_l(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    import torchsummary
    model = Colorizer()
    torchsummary.summary(model, (1, 32, 32))

        