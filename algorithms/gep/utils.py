
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os


import numpy as np

from rdp_accountant import compute_rdp, get_privacy_spent

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
DEFAULT_SIGMA_MIN_BOUND = 0.01
DEFAULT_SIGMA_MAX_BOUND = 10
MAX_SIGMA = 2000

SIGMA_PRECISION = 0.01

def get_data_loader(dataset, batchsize):
    if(dataset == 'svhn'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.SVHN('./data',split='train', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=73257, shuffle=True, num_workers=0) #load full btach into memory, to concatenate with extra data

        extraset = torchvision.datasets.SVHN('./data',split='extra', download=True, transform=transform)
        extraloader = torch.utils.data.DataLoader(extraset, batch_size=531131, shuffle=True, num_workers=0) #load full btach into memory

        testset = torchvision.datasets.SVHN('./data',split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)
        return trainloader, extraloader, testloader, len(trainset)+len(extraset), len(testset)
    else:
        transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
        return trainloader, testloader, len(trainset), len(testset)

def get_sigma(sample_rate, epochs, target_eps, delta, rgp=True):
    orders = DEFAULT_ALPHAS

    eps = float("inf")
    sigma_min = DEFAULT_SIGMA_MIN_BOUND
    sigma_max = DEFAULT_SIGMA_MAX_BOUND
    steps = int(epochs/sample_rate)

    while eps > target_eps:
        sigma_max = 2*sigma_max
        if(rgp):
            rdp = compute_rdp(sample_rate, sigma_max, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(sample_rate, sigma_max, steps, orders)
        eps, _, _ = get_privacy_spent(orders=orders, rdp=rdp, target_delta=delta)
        if sigma_max > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while sigma_max - sigma_min > SIGMA_PRECISION:

        sigma = (sigma_min + sigma_max)/2
        if(rgp):
            rdp = compute_rdp(sample_rate, sigma, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(sample_rate, sigma, steps, orders)
        eps, _, _ = get_privacy_spent(orders=orders, rdp=rdp, target_delta=delta)

        if eps < target_eps:
            sigma_max = sigma
        else:
            sigma_min = sigma

    return sigma, eps

def restore_param(cur_state, state_dict):
    own_state = cur_state
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)

def sum_list_tensor(tensor_list, dim=0):
    return torch.sum(torch.cat(tensor_list, dim=dim), dim=dim)

def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


def checkpoint(net, acc, epoch, sess):
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'approx_error': net.gep.approx_error
    }
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + sess  + '.ckpt')

def adjust_learning_rate(optimizer, init_lr, epoch, all_epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    decay = 1.0
    if(epoch<all_epoch*0.5):
        decay = 1.
    elif(epoch<all_epoch*0.75):
        decay = 10.
    else:
        decay = 100.

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr / decay
    return init_lr / decay
