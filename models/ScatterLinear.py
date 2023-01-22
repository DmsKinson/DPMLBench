import torch.nn as nn

class ScatterLinear(nn.Module):
    def __init__(self, in_channels, hw_dims,  classes=10, **kwargs):
        super(ScatterLinear, self).__init__()
        self.K = in_channels
        self.h = hw_dims[0]
        self.w = hw_dims[1]
        self.fc = None
        self.norm = None
        self.build(classes=classes, **kwargs)

    def build(self, num_groups=4, classes=10):
        self.fc = nn.Linear(self.K * self.h * self.w, classes)
        self.norm = nn.GroupNorm(num_groups, self.K, affine=False)

    def forward(self, x):
        x = self.norm(x.view(-1, self.K, self.h, self.w))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x