import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding="same")
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same")
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding="same")
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(32, 3, kernel_size=(3, 3), padding="same")
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.deconv3(x)
        x = self.relu3(x)
        x = self.conv(x)
        x = self.sig(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
