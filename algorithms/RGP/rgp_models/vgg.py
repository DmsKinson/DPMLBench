from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
from RGP.rgp_models.resnet_cifar import LrkConv2d

from RGP.rgp_models.resnet_cifar import lrk_conv3x3,LinearBackwardHook, LinearFowardHook, lrk_conv3x3


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

num_group = 4

class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        actv: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        fc1 = nn.Linear(512 * 7 * 7, 4096,bias=False)
        fc2 = nn.Linear(4096, 4096, bias=False)
        fc3 = nn.Linear(4096, num_classes, bias=False)

        for fc in [fc1,fc2,fc3]:
            fc.register_forward_hook(LinearFowardHook)
            fc.register_full_backward_hook(LinearBackwardHook)

        self.classifier = nn.Sequential(OrderedDict([
                ('fc1',fc1),
                ('actv',actv()),
                ('drop',nn.Dropout()),
                ('fc2',fc2),
                ('actv',actv()),
                ('drop',nn.Dropout()),
                ('fc3',fc3),
            ])
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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



def make_layers(cfg: List[Union[str, int]], actv, rank, batch_norm: bool = False, in_channel=3) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = lrk_conv3x3(in_channel, v, rank=rank, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.GroupNorm(num_group,v), actv()]
            else:
                layers += [conv2d, actv()]
            in_channel = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, actv, rank, in_channel=3, **kwargs: Any) -> VGG:
    model = VGG(features=make_layers(cfgs[cfg], actv, rank=rank, batch_norm=batch_norm, in_channel=in_channel),actv=actv, **kwargs)
    return model


def vgg11(actv = nn.ReLU, rank=16, progress: bool = True,in_channel = 3, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, rank=rank, actv=actv, in_channel=in_channel, **kwargs)
