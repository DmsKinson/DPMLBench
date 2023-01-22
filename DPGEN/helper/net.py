import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from functools import partial
from . import get_sigmas
from .layers import *
from .normalization import get_normalization

class MultiClassifier(nn.Module):
    def __init__(self,in_channel):
        super(MultiClassifier, self).__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3), # 3, 256, 256
            nn.MaxPool2d(2), # op: 16, 127, 127
            nn.ReLU(), # op: 64, 127, 127
        )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3), # 64, 127, 127   
            nn.MaxPool2d(2), #op: 128, 63, 63
            nn.ReLU() # op: 128, 63, 63
        )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3), # 128, 63, 63
            nn.MaxPool2d(2), #op: 256, 30, 30
            nn.ReLU() #op: 256, 30, 30
        )
        
        self.Linear1 = nn.Linear(1024, 64)
        self.Linear2 = nn.Linear(64, 10)
                
        
    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)

        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        x = self.Linear2(x)
        return x

class RefineNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False)
        self.ngf = ngf = config.model.ngf
        self.num_classes = num_classes = config.model.num_classes

        self.act = act = get_act(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)

        self.normalizer = self.norm(ngf, self.num_classes)
        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        if config.data.image_size == 28:
            self.res4 = nn.ModuleList([
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                              normalization=self.norm, adjust_padding=True, dilation=4),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                              normalization=self.norm, dilation=4)]
            )
        else:
            self.res4 = nn.ModuleList([
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                              normalization=self.norm, adjust_padding=False, dilation=4),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                              normalization=self.norm, dilation=4)]
            )

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output
