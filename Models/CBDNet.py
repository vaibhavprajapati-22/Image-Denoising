import torch
import torch.nn as nn


class Noise_Estimation(nn.Module):
    def __init__(self):
        super(Noise_Estimation, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding="same")
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding="same")
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding="same")
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, (3, 3), padding="same")
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(32, 3, (3, 3), padding="same")
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)

        return x


class Denoising(nn.Module):
    def __init__(self):
        super(Denoising, self).__init__()

        self.conv1 = nn.Conv2d(6, 64, (3, 3), padding="same")
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, (3, 3), padding="same")
        self.relu2 = nn.ReLU()

        self.pool1 = nn.AvgPool2d((2, 2), padding=0)

        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding="same")
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, (3, 3), padding="same")
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding="same")
        self.relu5 = nn.ReLU()

        self.pool2 = nn.AvgPool2d((2, 2), padding=0)

        self.conv6 = nn.Conv2d(128, 256, (3, 3), padding="same")
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256, 256, (3, 3), padding="same")
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 256, (3, 3), padding="same")
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(256, 256, (3, 3), padding="same")
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(256, 256, (3, 3), padding="same")
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(256, 256, (3, 3), padding="same")
        self.relu11 = nn.ReLU()

        self.convT1 = nn.ConvTranspose2d(256, 128, (2, 2), stride=2)
        self.reluT1 = nn.ReLU()

        self.conv12 = nn.Conv2d(128, 128, (3, 3), padding="same")
        self.relu12 = nn.ReLU()
        self.conv13 = nn.Conv2d(128, 128, (3, 3), padding="same")
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(128, 128, (3, 3), padding="same")
        self.relu14 = nn.ReLU()

        self.convT2 = nn.ConvTranspose2d(128, 64, (2, 2), stride=2)
        self.reluT2 = nn.ReLU()

        self.conv15 = nn.Conv2d(64, 64, (3, 3), padding="same")
        self.relu15 = nn.ReLU()
        self.conv16 = nn.Conv2d(64, 64, (3, 3), padding="same")
        self.relu16 = nn.ReLU()
        self.conv17 = nn.Conv2d(64, 3, (3, 3), padding="same")
        self.relu17 = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu1(x1)
        x3 = self.conv2(x2)
        x4 = self.relu2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv3(x5)
        x7 = self.relu3(x6)
        x8 = self.conv4(x7)
        x9 = self.relu4(x8)
        x10 = self.conv5(x9)
        x11 = self.relu5(x10)
        x12 = self.pool2(x11)
        x13 = self.conv6(x12)
        x14 = self.relu6(x13)
        x15 = self.conv7(x14)
        x16 = self.relu7(x15)
        x17 = self.conv8(x16)
        x18 = self.relu8(x17)
        x19 = self.conv9(x18)
        x20 = self.relu9(x19)
        x21 = self.conv10(x20)
        x22 = self.relu10(x21)
        x23 = self.conv11(x22)
        x24 = self.relu11(x23)
        x25 = self.convT1(x24)
        x26 = self.reluT1(x25)
        x27 = x26 + x11
        x28 = self.conv12(x27)
        x29 = self.relu12(x28)
        x30 = self.conv13(x29)
        x31 = self.relu13(x30)
        x32 = self.conv14(x31)
        x33 = self.relu14(x32)
        x34 = self.convT2(x33)
        x35 = self.reluT2(x34)
        x36 = x35 + x4
        x37 = self.conv15(x36)
        x38 = self.relu15(x37)
        x39 = self.conv16(x38)
        x40 = self.relu16(x39)
        x41 = self.conv17(x40)
        x42 = self.relu17(x41)

        return x42


class CBDNet(nn.Module):
    def __init__(self):
        super(CBDNet, self).__init__()

        self.noise_estimation = Noise_Estimation()
        self.denoising = Denoising()

    def forward(self, input):
        x = self.noise_estimation(input)
        x = torch.cat([x, input], dim=1)
        x = self.denoising(x)
        x = x + input
        return x