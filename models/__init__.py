from typing import Literal
from .resnet import resnet110, resnet1202, resnet20, resnet32, resnet44,resnet74,resnet8102
from .simple import SimpleNN, SimpleNN_Norm, simpleNet
from .vgg import vgg11
from .alexnet import AlexNet
from .inceptionSE import SimpleInception
from .attack_white_box import WhiteBoxAttackModel, WhiteBoxAttackModelBinary
from .attack_black_box import ShadowAttackModel
import torchvision
import torch.nn as nn

def get_model(name:str, dataset=None, act_func='relu', in_channel=1, **kwargs):
    if(in_channel in (1,3) ):
        in_channel = 1 if dataset in ['mnist','fmnist'] else 3

    actv = nn.ReLU if act_func.lower()=='relu' else nn.Tanh

    if name.lower() == 'simple':
        return simpleNet(in_channel=in_channel, actv=actv, **kwargs)
    elif name.lower() == 'resnet':
        return resnet20(in_channel=in_channel, actv=actv, **kwargs)
    elif name.lower() == 'resnet32':
        return resnet32(in_channel=in_channel, actv=actv, **kwargs)
    elif name.lower() == 'resnet44':
        return resnet44(in_channel=in_channel, actv=actv, **kwargs)
    elif name.lower() == 'resnet74':
        return resnet74(in_channel=in_channel, actv=actv, **kwargs)
    elif name.lower() == 'resnet110':
        return resnet110(in_channel=in_channel, actv=actv, **kwargs)
    elif name.lower() == 'resnet1202':
        return resnet1202(in_channel=in_channel, actv=actv, **kwargs)
    elif name.lower() == 'resnet8102':
        return resnet8102(in_channel=in_channel, actv=actv, **kwargs)
    elif name.lower() == 'simplenn':
        return SimpleNN(in_channel=in_channel, actv=actv, **kwargs)
    elif name.lower() == 'simplenn_norm':
        return SimpleNN_Norm(in_channel=in_channel, actv=actv, **kwargs)
    elif name.lower() == 'vgg':
        return vgg11(in_channel=in_channel, actv=actv, num_classes=10, **kwargs)
    elif name.lower() == 'alexnet':
        return AlexNet(in_channel=in_channel, actv=actv, num_classes=10)
    elif name.lower() == 'googlenet':
        return torchvision.models.googlenet(num_classes=10)
    elif name.lower() == 'densenet':
        return torchvision.models.densenet121(num_classes=10)
    elif name.lower() == 'inception':
        return SimpleInception(in_channel=in_channel, actv=actv, **kwargs)
    elif name.lower() == 'full_inception':
        return torchvision.models.inception_v3(num_classes=10, aux_logits=False)
    else:
        raise Exception(f"illegal net:{name}") 

def get_attack_model(type:Literal['white','black','label','white_test'], **kwargs):
    num_class = 10
    if('black' == type):
        return ShadowAttackModel(class_num=num_class)
    elif('white' == type):
        return WhiteBoxAttackModel(num_class, total=kwargs['total'])
    elif('label' == type):
        return None
    elif('white_test' == type):
        return WhiteBoxAttackModelBinary(num_class, kernel_size=kwargs['kernel_size'], layer_size=kwargs['layer_size'])
    else:
        raise NotImplementedError
