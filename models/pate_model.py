import torch
import torch.nn as nn

# Using pytorch to implement the model structure in the PATE origin paper

class pate_model(nn.Module):
    def __init__(self, in_channel) -> None:
        super(pate_model, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel,64,5,1,padding=2),
            nn.MaxPool2d(3,2),
            nn.BatchNorm2d()
        )

    def forward(self,x):
        pass