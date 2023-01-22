import torch
import torch.nn as nn

class simpleNet(nn.Module):
    def __init__(self, actv, in_channel=1, **kwargs):
        super().__init__()
        # self.handcraft = handcraft
        # if(self.handcraft):
        #     self.scattering = Scattering2D(2,[32*4,32*4])
        #     in_channel *= 81
        self.in_channel = in_channel
        self.actv = actv()
        self.conv1 = nn.Conv2d(in_channel,16,3,padding = 1)
        self.conv2 = nn.Conv2d(16,32,3,padding = 1)
        self.conv3 = nn.Conv2d(32,64,3,padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.h1 = nn.Linear(64*4*4,500)
        self.h2 = nn.Linear(500,10)

    def forward(self,x,feature = False):
        # x :: bs*channel*w*h : data
        # matching :: bool : whether return feature
        # if(hasattr(self,'handcraft') and self.handcraft):
        #     with torch.no_grad():
        #         x = self.scattering(x).view(-1,self.in_channel,32,32)
        x = self.pool(self.actv(self.conv1(x)))
        x = self.pool(self.actv(self.conv2(x)))
        x = self.pool(self.actv(self.conv3(x)))
        x = torch.flatten(x,1)

        x = self.actv(self.h1(x))
        x = self.h2(x)
        return x

class SimpleNN(nn.Module):
    def __init__(self, actv, in_channel=1, **kwargs) -> None:
        super().__init__()
        # self.handcraft = handcraft
        # if(self.handcraft):
        #     self.scattering = Scattering2D(2,[32*4,32*4])
        #     in_channel *= 81
        self.in_channel = in_channel
        self.actv = actv()
        self.conv1 = nn.Conv2d(in_channel,16,3,padding = 1)
        self.conv2 = nn.Conv2d(16,32,3,padding = 1)
        self.conv3 = nn.Conv2d(32,32,3,padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.h1 = nn.Linear(32*4*4,300)
        self.h2 = nn.Linear(300,10)

    def forward(self, x, feature=False):
        # if(hasattr(self,'handcraft') and self.handcraft):
        #     with torch.no_grad():
        #         x = self.scattering(x).view(-1,self.in_channel,32,32)
        x = self.pool(self.actv(self.conv1(x)))
        x = self.pool(self.actv(self.conv2(x)))
        x = self.pool(self.actv(self.conv3(x)))
        x = torch.flatten(x,1)
        if(feature==True):
            return x
        x = self.actv(self.h1(x))
        x = self.h2(x)
        return x

class SimpleNN_Norm(nn.Module):
    def __init__(self, actv, in_channel=1, **kwargs) -> None:
        super().__init__()
        # self.handcraft = handcraft
        # if(self.handcraft):
        #     self.scattering = Scattering2D(2,[32*4,32*4])
        #     in_channel *= 81
        self.in_channel = in_channel
        self.actv = actv()
        self.conv1 = nn.Conv2d(in_channel,16,3,padding = 1)
        self.norm1 = nn.GroupNorm(4,16,affine=False)

        self.conv2 = nn.Conv2d(16,32,3,padding = 1)
        self.norm2 = nn.GroupNorm(4,32,affine=False)

        self.conv3 = nn.Conv2d(32,32,3,padding = 1)
        self.norm3 = nn.GroupNorm(4,32,affine=False)

        self.pool = nn.MaxPool2d(2,2)
        self.h1 = nn.Linear(32*4*4,300)
        self.h2 = nn.Linear(300,10)

    def forward(self, x):
        # if(hasattr(self,'handcraft') and self.handcraft):
        #     with torch.no_grad():
        #         x = self.scattering(x).view(-1,self.in_channel,32,32)
        x = self.pool(self.actv(self.norm1(self.conv1(x))))
        x = self.pool(self.actv(self.norm2(self.conv2(x))))
        x = self.pool(self.actv(self.norm3(self.conv3(x))))
        x = torch.flatten(x,1)
        x = self.actv(self.h1(x))
        x = self.h2(x)
        return x

class print_shape(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        print(x.shape)
        return x

class mnistTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.actv = nn.Tanh()
        self.feature = nn.Sequential(
            nn.Conv2d(1,16,8,2),
            self.actv,
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,4,2),
            self.actv,
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32,10)
        )
    def forward(self,x):
        x = self.feature(x)
        return x