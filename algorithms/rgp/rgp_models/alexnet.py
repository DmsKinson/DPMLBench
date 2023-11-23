import torch.nn as nn
import torch

from .resnet_cifar import LinearBackwardHook, LinearFowardHook, LrkConv2d

class AlexNet(nn.Module):

    def __init__(self, actv = nn.ReLU, rank = 16, in_channel=3, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            LrkConv2d(in_channel, 64, kernel_size=11, stride=4, padding=2, rank=rank),
            actv(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LrkConv2d(64, 192, kernel_size=5, padding=2, rank=rank),
            actv(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LrkConv2d(192, 384, kernel_size=3, padding=1, rank=rank),
            actv(),
            LrkConv2d(384, 256, kernel_size=3, padding=1, rank=rank),
            actv(),
            LrkConv2d(256, 256, kernel_size=3, padding=1, rank=rank),
            actv(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        fc1 = nn.Linear(256*6*6, 4096)
        fc2 = nn.Linear(4096, 4096)
        fc3 = nn.Linear(4096, num_classes)

        for fc in [fc1, fc2, fc3]:
            fc.register_full_backward_hook(LinearBackwardHook)
            fc.register_forward_hook(LinearFowardHook)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            fc1,
            actv(),
            nn.Dropout(),
            fc2,
            actv(),
            fc3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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

