"""
Formulate All Your Network Models Here
"""

from lib import *

class MLP(nn.Module):
    def __init__(self, in_size:int=3*32*32):
        """
        MultiLayer Perceptron model
        """
        super(MLP, self).__init__()
        self.in_size = in_size
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        logits = self.linear_relu_stack(x)
        return logits
    
    def __str__(self):
        return f'in_size: {self.in_size}\n\n'+super().__str__()
    
class Conv(nn.Module):
    def __init__(self, n_channels:int=3):
        super(Conv, self).__init__()
        self.n_channels = n_channels
        # self.flatten = nn.Flatten()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, 7, 2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1024, 3, 2, 1),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        x = self.gap(x)
        x = self.flatten(x)
        # print(x.shape)
        logits = self.linear(x)
        return logits
    
    def __str__(self):
        return f'n_channels: {self.n_channels}\n\n'+super().__str__()
    
import torch.nn.functional as F

class _ResBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out
    
class _BigResBlock(nn.Module):

    def __init__(self, inchannel, midchannel, outchannel, stride=1):
        super(_BigResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel,midchannel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(midchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(midchannel,midchannel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(midchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(midchannel,outchannel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out
    
class ResNet18(nn.Module):

    def __init__(self, ResBlock=_ResBlock, num_classes=10):
        super(ResNet18, self).__init__()

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU()
        )
        self.lay1 = self.make_lay(ResBlock, 64, 2, stride=2)
        self.lay2 = self.make_lay(ResBlock, 128, 2, stride=2)
        self.lay3 = self.make_lay(ResBlock, 256, 2, stride=2)
        self.lay4 = self.make_lay(ResBlock, 512, 2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
        
        
    def make_lay(self, block=_ResBlock, channels=64, num_blocks=2, stride=1):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lay1(out)
        out = self.lay2(out)
        out = self.lay3(out)
        out = self.lay4(out)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)

        return out
    
class ResNet34(nn.Module):

    def __init__(self, ResBlock=_ResBlock, in_channel=3, num_classes=10, first_lay_kernel_size=5, first_lay_stride=1, first_lay_padding=2):
        super(ResNet34, self).__init__()

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, self.inchannel, kernel_size=first_lay_kernel_size, 
                      stride=first_lay_stride, padding=first_lay_padding, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU()
        )
        self.lay1 = self.make_lay(ResBlock, 64, 3, stride=2)
        self.lay2 = self.make_lay(ResBlock, 128, 4, stride=2)
        self.lay3 = self.make_lay(ResBlock, 256, 6, stride=2)
        self.lay4 = self.make_lay(ResBlock, 512, 3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
        
        
    def make_lay(self, block=_ResBlock, channels=64, num_blocks=2, stride=1):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lay1(out)
        out = self.lay2(out)
        out = self.lay3(out)
        out = self.lay4(out)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)

        return out
    
class ResNet50(nn.Module):

    def __init__(self, ResBlock=_BigResBlock, num_classes=10):
        super(ResNet50, self).__init__()

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU()
        )
        self.lay1 = self.make_lay(ResBlock, 64, 3, stride=2)
        self.lay2 = self.make_lay(ResBlock, 128, 4, stride=2)
        self.lay3 = self.make_lay(ResBlock, 256, 23, stride=2)
        self.lay4 = self.make_lay(ResBlock, 512, 3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(self.inchannel, num_classes)
        
        
    def make_lay(self, block=_BigResBlock, channels=64, num_blocks=2, stride=1):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, channels*4, stride))
            self.inchannel = channels*4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lay1(out)
        out = self.lay2(out)
        out = self.lay3(out)
        out = self.lay4(out)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)

        return out
