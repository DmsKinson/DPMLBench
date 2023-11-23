from .resnet_cifar import resnet28
from .simple import SimpleNN, simpleNet
from .vgg import vgg11
from .alexnet import AlexNet
from .inception import SimpleInception


def get_model(name:str, rank, dataset=None, in_channel=1):
    
    if(in_channel in (1,3) ):
        in_channel = 1 if dataset in ['mnist','fmnist'] else 3

    if name.lower() == 'simple':
        return simpleNet(in_channel=in_channel, rank=rank)
    elif name.lower() == 'simplenn':
        return SimpleNN(in_channel=in_channel, rank=rank)
    elif name.lower() == 'resnet':
        return resnet28(in_channel=in_channel, num_classes=10, rank=rank)
    elif name.lower() == 'vgg':
        return vgg11(in_channel=in_channel, num_classes=10, rank=rank)
    elif name.lower() == 'inception':
        return  SimpleInception(in_channel=in_channel,rank=rank)
    else:
        raise Exception(f"illegal net:{name}") 
