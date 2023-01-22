import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from DataFactory import SUPPORTED, DataFactory

class _basic_conv(nn.Module):
        def __init__(self, in_channels, out_channels, **kwargs):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

class _block(nn.Module):
    def __init__(self, in_channels, c1, c2):
        super().__init__()
        self.conv1 = _basic_conv(in_channels, c1, kernel_size=1)
        self.conv2 = _basic_conv(in_channels, c2, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return torch.cat((x1,x2),dim=1)

class conv_mnist(nn.Module):
    class S1(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = _block(in_channels=96, c1=32, c2=32)
            self.block2 = _block(in_channels=64, c1=32, c2=48)
            self.conv = _basic_conv(in_channels=80, out_channels=160, kernel_size=3, stride=2)
        
        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.conv(x)
            return x

    class S2(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = _block(in_channels=160, c1=112, c2=48)
            self.block2 = _block(in_channels=160, c1=96, c2=64)
            self.block3 = _block(in_channels=160, c1=80, c2=80)
            self.block4 = _block(in_channels=160, c1=48, c2=96)
            self.conv = _basic_conv(in_channels=144, out_channels=240, kernel_size=3, stride=2)

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.conv(x)
            return x

    class S3(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = _block(in_channels=240, c1=176, c2=160)
            self.block2 = _block(in_channels=336, c1=176, c2=160)
        
        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            return x
            
    
    def __init__(self):
        super().__init__()
        self.conv = _basic_conv(in_channels=1, out_channels=96, kernel_size=3)
        self.s1 = self.S1()
        self.s2 = self.S2()
        self.s3 = self.S3()
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=1)
        self.fc = nn.Linear(336*4*4, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.max_pool(x)
        x = x.view(-1, 336*4*4)
        x = self.fc(x)
        return x

def get_model(dataset:str):
    assert dataset.lower() in SUPPORTED , "Unsupported dataset. Select from [mnist,cifar10]"
    if dataset.lower() == 'mnist':
        return conv_mnist()
    elif dataset.lower():
        return torchvision.models.resnet18(pretrained=False)
            