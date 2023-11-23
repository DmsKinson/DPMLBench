import torch.nn as nn

class netD(nn.Module):
    def __init__(self, nc, ndf):
        super(netD, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
        )
        self.main2 = nn.Sequential(
            nn.Linear(1024,10),
        )
        self.main3 = nn.Sequential(
            nn.Sigmoid()
        )
    def forward(self, input,matching = False):
        output = self.main(input)
        feature = output.view(-1,1024)
        output = self.main2(feature)
        #output = self.main3(output)
        if matching == True:
            return feature
        else:
            return output #batch_size x 1 x 1 x 1 => batch_size