import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, layers=1):
        super(ConvBlock, self).__init__()
        padding = (kernel_size[0] - 1) // 2
        # padding = ((stride[0] - 1) * in_channels - 1 + kernel_size[0]) // 2
        self.initial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(padding,0))

        # conv blocks
        self.conv_layers = nn.ModuleList()
        for _ in range(layers):
            layer = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(padding,0)),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(padding,0)),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride,padding=(padding,0))
            )
            self.conv_layers.append(layer)

    def forward(self, x):
        x = self.initial_conv(x)

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)

        return x
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, layers=4):
        super(ResNetBlock, self).__init__()
        padding = (kernel_size[0] - 1) // 2
        # padding = ((stride[0] - 1) * in_channels - 1 + kernel_size[0]) // 2
        self.initial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(padding,0))

        # Define the first set of conv layers and pooling
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(padding,0))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(padding,0))
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,padding=(padding,0))

        # Residual blocks
        self.res_layers = nn.ModuleList()
        for _ in range(layers):
            layer = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(padding,0)),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(padding,0)),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride,padding=(padding,0))
            )
            self.res_layers.append(layer)

        # Shortcut connection for residual blocks
        self.shortcut_convs = nn.ModuleList()
        for _ in range(layers):
            shortcut = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(padding,0)),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride,padding=(padding,0))
            )
            self.shortcut_convs.append(shortcut)

    def forward(self, x):
        x = self.initial_conv(x)

        # First set of conv layers and pooling
        x_short = self.maxpool(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = x + x_short  # Add shortcut

        # Residual blocks
        for i in range(len(self.res_layers)):
            x_short = self.shortcut_convs[i](x)
            x = self.res_layers[i](x)
            x = x + x_short  # Add shortcut

        return x

class AFNet(nn.Module):
    def __init__(self):
        super(AFNet, self).__init__()
        inchannels = 1
        self.resnet_block1 = ResNetBlock(1, 4, (5, 1), (1, 1))
        self.resnet_block2 = ResNetBlock(1, 4, (11, 1), (1, 1))
        self.resnet_block3 = ResNetBlock(1, 4, (15, 1), (1, 1))

        self.conv1 = ConvBlock(4, 1, (5, 1), (1, 1))
        self.conv2 = ConvBlock(4, 1, (5, 1), (1, 1))
        self.conv3 = ConvBlock(4, 1, (5, 1), (1, 1))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1250, out_features=10)
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, input):

        x1 = self.resnet_block1(input)
        x1 = self.conv1(x1)
        x2 = self.resnet_block2(input)
        x2 = self.conv2(x2)
        x3 = self.resnet_block3(input)
        x3 = self.conv3(x3)

        x = x1 + x2 + x3

        x = x.view(-1, 1250)

        fc1_output = F.relu(self.fc1(x))
        fc2_output = self.fc2(fc1_output)
        return fc2_output

