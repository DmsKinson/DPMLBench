from typing import Callable
import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
from .resnet_cifar import LinearBackwardHook, LinearFowardHook, lrk_conv3x3

class simpleNet(nn.Module):
    def __init__(self, in_channel=1, rank=16):
        super().__init__()
        self.actv = nn.ReLU(inplace=False)
        self.conv1 = lrk_conv3x3(in_planes=in_channel, out_planes=16, kernel_size=3, padding = 1, rank=rank)
        self.conv2 = lrk_conv3x3(in_planes=16, out_planes=32, kernel_size=3, padding = 1, rank=rank)
        self.conv3 = lrk_conv3x3(in_planes=32, out_planes=64, kernel_size=3, padding = 1, rank=rank)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*4*4,500, bias=False)
        self.fc2 = nn.Linear(500,10, bias=False)

        self.fc1.register_forward_hook(LinearFowardHook)
        self.fc1.register_backward_hook(LinearBackwardHook)
        self.fc2.register_backward_hook(LinearBackwardHook)
        self.fc2.register_forward_hook(LinearFowardHook)

    def forward(self,x):
        x = self.pool(self.actv(self.conv1(x)))
        x = self.pool(self.actv(self.conv2(x)))
        x = self.pool(self.actv(self.conv3(x)))
        x = torch.flatten(x,1)
        x = self.actv(self.fc1(x))
        x = self.fc2(x)
        return x

    def update_weight(self):
        for m in self.modules():
            if(hasattr(m, '_update_weight')):
                m._update_weight()

    def update_init_weight(self):
        for m in self.modules():
            if(hasattr(m, '_update_init_weight')):
                m._update_init_weight()

    def decomposite_weight(self):
        for m in self.modules():
            if(hasattr(m, '_decomposite_weight')):
                m._decomposite_weight()

class SimpleNN(nn.Module):
    def __init__(self, in_channel=1, rank=16) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.actv = nn.ReLU(inplace=False)
        self.conv1 = lrk_conv3x3(in_planes=in_channel, out_planes=16, kernel_size=3, padding = 1, rank=rank)
        self.conv2 = lrk_conv3x3(in_planes=16, out_planes=32, kernel_size=3, padding = 1, rank=rank)
        self.conv3 = lrk_conv3x3(in_planes=32, out_planes=32, kernel_size=3, padding = 1, rank=rank)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*4*4,300, bias=False)
        self.fc2 = nn.Linear(300,10, bias=False)

        self.fc1.register_forward_hook(LinearFowardHook)
        self.fc1.register_backward_hook(LinearBackwardHook)
        self.fc2.register_backward_hook(LinearBackwardHook)
        self.fc2.register_forward_hook(LinearFowardHook)

    def forward(self, x):
        x = self.pool(self.actv(self.conv1(x)))
        x = self.pool(self.actv(self.conv2(x)))
        x = self.pool(self.actv(self.conv3(x)))
        x = torch.flatten(x,1)
        x = self.actv(self.fc1(x))
        x = self.fc2(x)
        return x

    def update_weight(self):
        for m in self.modules():
            if(hasattr(m, '_update_weight')):
                m._update_weight()

    def update_init_weight(self):
        for m in self.modules():
            if(hasattr(m, '_update_init_weight')):
                m._update_init_weight()

    def decomposite_weight(self):
        for m in self.modules():
            if(hasattr(m, '_decomposite_weight')):
                m._decomposite_weight()
    
