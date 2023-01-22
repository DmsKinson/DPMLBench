from typing import Iterable, Literal
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataset import Subset
import torch

import os
pwd = os.path.split(os.path.realpath(__file__))[0]

SUPPORTED = ['mnist','fmnist','imagenet', 'svhn', 'cifar10','cifar100']
PUBSET_LENGTH = 0
WHOLE_TEST_LENGTH = 10000       # trim all testset to the same

UDA_TEST_LENGTH = 1000  # UDA testset length of target model

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
MNIST_TRANS = [
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
        transforms.Resize(32)
    ]

FMNIST_MEAN = (0.286,)
FMNIST_STD = (0.320,)
FMNIST_TRANS = [
        transforms.ToTensor(),
        transforms.Normalize(FMNIST_MEAN, FMNIST_STD),
        transforms.Resize(32)
    ]

SVHN_MEAN = (0.43768218, 0.44376934, 0.47280428)
SVHN_STD = (0.1980301, 0.2010157, 0.19703591)
SVHN_TRANS = [
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN,SVHN_STD),
    ]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_TRANS = [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)
    ]

CIFAR10_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR10_STD = (0.24703233, 0.24348505, 0.26158768)
CIFAR10_TRANS = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD),
    ]

CIFAR100_MEAN = (0.50707515, 0.48654887, 0.44091784)
CIFAR100_STD = (0.26733428, 0.25643846, 0.27615047)
CIFAR100_TRANS = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN,CIFAR100_STD),
    ]

TRANSFORM_DICT = {
    'mnist':MNIST_TRANS,
    'fmnist':FMNIST_TRANS,
    'cifar10':CIFAR10_TRANS,
    'cifar100':CIFAR100_TRANS,
    'svhn':SVHN_TRANS,
}

class DataFactory():
    def __init__(self, which:str, data_root=os.path.join(pwd,'dataset')):
        self.which = which.lower()
        assert self.which in SUPPORTED
        self.transform = TRANSFORM_DICT[self.which]
        self.dataRoot = data_root
        
    def _getTestSet(self, transform_list:Iterable = None):        
        if(transform_list == None):
            trans_comp = transforms.Compose(self.transform)
        else:
            trans_comp = transforms.Compose(transform_list)

        if self.which == 'mnist':
            return datasets.MNIST(
                self.dataRoot,
                train=False,
                download=True,
                transform= trans_comp,
            )
        elif self.which == "svhn":
            return datasets.SVHN(
                root = self.dataRoot,
                split="test",
                download=True,
                transform= trans_comp,
            )
        elif self.which == 'cifar10':
            return datasets.CIFAR10(
                root=self.dataRoot,
                train=False,
                transform= trans_comp,
                download=True
            )
        elif self.which == 'cifar100':
            return datasets.CIFAR100(
                root=self.dataRoot,
                train=False,
                transform=trans_comp,
                download=True
            )
        elif self.which == 'imagenet':
            return datasets.ImageNet(
                root = self.dataRoot,
                split="val",
                download=True,
                transform= trans_comp,
            )
        elif self.which == 'fmnist':
            return datasets.FashionMNIST(
                root = self.dataRoot,
                train=False,
                download=True,
                transform = trans_comp,
            )

    def _getTrainSet(self, transform_list:Iterable = None):
        if(transform_list == None):
            trans_comp = transforms.Compose(self.transform)
        else:
            trans_comp = transforms.Compose(transform_list)
        if self.which == 'mnist':
            return datasets.MNIST(
                self.dataRoot,
                train=True,
                download=True,
                transform= trans_comp,
            )
        elif self.which == 'svhn':
            return datasets.SVHN(
                root = self.dataRoot,
                split="train",
                download=True,
                transform= trans_comp,
            )
        elif self.which == 'cifar10':
            return datasets.CIFAR10(
                root=self.dataRoot,
                train=True,
                transform=trans_comp,
                download=True
            )
        elif self.which == 'cifar100':
            return datasets.CIFAR100(
                root=self.dataRoot,
                train=True,
                transform=trans_comp,
                download=True
            )
        elif self.which == 'imagenet':
            return datasets.ImageNet(
                root = self.dataRoot,
                split="train",
                download=True,
                transform=trans_comp,
            )
        elif self.which == 'fmnist':
            return datasets.FashionMNIST(
                root=self.dataRoot,
                train=True,
                transform=trans_comp,
                download=True
            )

    def getPubSet(self, transform_list:Iterable = None):
        assert PUBSET_LENGTH != 0
        pubset = self._getTestSet(transform_list)
        indices = list(range(len(pubset)))
        pubset = Subset(pubset, indices[:PUBSET_LENGTH])
        return pubset

    def getUdaTrainset(self, mode:Literal['target','shadow'] = 'target', size:int = None, transform_list:Iterable = None):
        mode = mode.lower()
        if(mode == 'target'):
            trainset = self.getTrainSet('target')
        elif(mode == 'shadow'):
            # split origin shadow dataset to uda trainset and uda testset
            trainset = self.getTrainSet('shadow')
            indices = list(range(len(trainset)))
            trainset = Subset(trainset, indices[UDA_TEST_LENGTH:])
        else:
            raise NotImplementedError
        return trainset

    def getUdaTestset(self, mode:Literal['target','shadow'] = 'target', size:int = None, transform_list:Iterable = None):
        mode = mode.lower()
        if(mode == 'target'):
            testset = self.getTestSet('target')
        elif(mode == 'shadow'):
            # split origin shadow dataset to uda trainset and uda testset
            testset = self.getTrainSet('shadow')
            indices = list(range(len(testset)))
            testset = Subset(testset, indices[:UDA_TEST_LENGTH])
        else:
            raise NotImplementedError
        return testset

    def getTestSet(self, mode:Literal['target','shadow','full','custom'] = 'target', size:int = None, transform_list:Iterable = None):
        mode = mode.lower()
        assert not (mode == 'custom' and size == None) , 'custom mode should provide size parameter'           
        testset = self._getTestSet(transform_list)
        whole_len = min(WHOLE_TEST_LENGTH,len(testset))
        indices = list(range(whole_len))
        test_len = whole_len//2

        if(mode == 'target'):
            testset = Subset(testset, indices[:test_len])
        elif(mode == 'shadow'):
            testset = Subset(testset, indices[test_len:])
        elif(mode == 'custom'):
            testset = Subset(testset, indices[:size])
        elif(mode == 'full'):
            testset = testset
        else:
            raise NotImplementedError
        return testset

    def getTrainSet(self, mode:Literal['target','shadow','full','custom'] = 'target', size:int = None, transform_list:Iterable = None):
        mode = mode.lower()
        assert not (mode == 'custom' and size == None) , 'custom mode should provide size parameter'           
        trainset = self._getTrainSet(transform_list)
        indices = list(range(len(trainset)))
        if(mode == 'target'):
            trainset = Subset(trainset, indices[:len(trainset)//2])
        elif(mode == 'shadow'):
            trainset = Subset(trainset, indices[len(trainset)//2:])
        elif(mode == 'custom'):
            trainset = Subset(trainset, indices[:size])
        elif(mode == 'full'):
            trainset = trainset
        else:
            raise NotImplementedError
        return trainset

