import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from rgp_models.resnet_cifar import LrkConv2d,LinearBackwardHook, LinearFowardHook


from typing import *

local_actv = nn.ReLU
local_rank = 16
NUM_GROUP = 4

class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        global local_rank
        # set rank to 16 defaultly
        self.conv = LrkConv2d(in_channels, out_channels, rank=local_rank, **kwargs)
        self.bn = nn.GroupNorm(NUM_GROUP,out_channels, eps=0.001)
        self.actv = local_actv()

    def forward(self, x) :
        x = self.conv(x)
        x = self.bn(x)
        return self.actv(x)


class InceptionA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 32, kernel_size=1, padding=0)

        self.branch5x5_1 = conv_block(in_channels, 24, kernel_size=1, padding=0)
        self.branch5x5_2 = conv_block(24, 32, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1, padding=0)
        self.branch3x3dbl_2 = conv_block(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(48, 48, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1, padding=0)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 96, kernel_size=3, stride=2, padding=0)

        self.branch3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1, padding=0)
        self.branch3x3dbl_2 = conv_block(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(48, 48, kernel_size=3, stride=2, padding=0)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionC(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 48, kernel_size=1, padding=0)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1, padding=0)
        self.branch7x7_2 = conv_block(c7, 48, kernel_size=7, padding=3)

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1, padding=0)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=7, padding=3)
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=7, padding=3)


        self.branch_pool = conv_block(in_channels, 48, kernel_size=1, padding=0)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionD(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 48, kernel_size=1, padding=0)
        self.branch3x3_2 = conv_block(48, 96, kernel_size=3, stride=2, padding=0)

        self.branch7x7x3_1 = conv_block(in_channels, 96, kernel_size=1, padding=0)
        self.branch7x7x3_2 = conv_block(96, 96, kernel_size=7, padding=3)
        self.branch7x7x3_3 = conv_block(96, 96, kernel_size=3, stride=2, padding=0)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 80, kernel_size=1, padding=0)

        self.branch3x3_1 = conv_block(in_channels, 96, kernel_size=1, padding=0)
        self.branch3x3_2 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch3x3dbl_1 = conv_block(in_channels, 112, kernel_size=1, padding=0)
        self.branch3x3dbl_2 = conv_block(112, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, 48, kernel_size=1, padding=0)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, dim=1)

class SimpleInception(nn.Module):

    def __init__(self, in_channel=1, actv=nn.ReLU, rank=1):
        super().__init__()
        global local_actv, local_rank
        local_actv = actv
        local_rank = rank
        self.feature = nn.Sequential(
            BasicConv2d(in_channel, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2,1),
            InceptionA(32, 16),
            InceptionB(128),
            InceptionC(272,48),
            InceptionD(192),
            InceptionE(384)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(320, 10, bias=False)
        self.fc.register_forward_hook(LinearFowardHook)
        self.fc.register_full_backward_hook(LinearBackwardHook)


    def forward(self, x):
        x = self.feature(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
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

        
            