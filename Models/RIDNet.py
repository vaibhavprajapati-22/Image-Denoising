import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels // reduction, 1, 1, 0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels // reduction, in_channels, 1, 1, 0)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        gap = self.gap(x)
        x_out = self.conv1(gap)
        x_out = self.relu1(x_out)
        x_out = self.conv2(x_out)
        x_out = self.sigmoid2(x_out)
        x_out = x_out * x
        return x_out


class EAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, reduciton=4):
        super(EAM, self).__init__()

        self.path1_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        self.path1_relu1 = nn.ReLU()
        self.path1_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=2)
        self.path1_relu2 = nn.ReLU()

        self.path2_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=3, dilation=3)
        self.path2_relu1 = nn.ReLU()
        self.path2_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=4, dilation=4)
        self.path2_relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels * 2, out_channels, kernel_size, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu8 = nn.ReLU()

        self.ca = ChannelAttention(in_channels, out_channels, reduction=16)

    def forward(self, x):
        x1 = self.path1_conv1(x)
        x1 = self.path1_relu1(x1)
        x1 = self.path1_conv2(x1)
        x1 = self.path1_relu2(x1)

        x2 = self.path2_conv1(x)
        x2 = self.path2_relu1(x2)
        x2 = self.path2_conv2(x2)
        x2 = self.path2_relu2(x2)

        x3 = torch.cat([x1, x2], dim=1)
        x3 = self.conv3(x3)
        x3 = self.relu3(x3)
        x3 = x3 + x

        x4 = self.conv4(x3)
        x4 = self.relu4(x4)
        x4 = self.conv5(x4)
        x5 = x4 + x3
        x5 = self.relu5(x5)

        x6 = self.conv6(x5)
        x6 = self.relu6(x6)
        x7 = self.conv7(x6)
        x7 = self.relu7(x7)
        x8 = self.conv8(x7)
        x8 = x8 + x5
        x8 = self.relu8(x8)

        x_ca = self.ca(x8)

        return x_ca + x


class RIDNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_feautres):
        super(RIDNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, num_feautres, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=False)

        self.eam1 = EAM(in_channels=num_feautres, out_channels=num_feautres)
        self.eam2 = EAM(in_channels=num_feautres, out_channels=num_feautres)
        self.eam3 = EAM(in_channels=num_feautres, out_channels=num_feautres)
        self.eam4 = EAM(in_channels=num_feautres, out_channels=num_feautres)

        self.last_conv = nn.Conv2d(num_feautres, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)

        self.init_weights()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu1(x1)

        x_eam = self.eam1(x1)
        x_eam = self.eam2(x_eam)
        x_eam = self.eam3(x_eam)
        x_eam = self.eam4(x_eam)

        x_lsc = x_eam + x1
        x_out = self.last_conv(x_lsc)
        x_out = x_out + x

        return x_out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)