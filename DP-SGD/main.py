import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import opacus
from tools import MemoryManagerProxy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from DataFactory import DataFactory
import tools

import argparse
import time
import sqlite_proxy
from data_manager import get_md5
from opacus.validators import ModuleValidator
from clip_only_utils import ClipOnlyPrivacyEngine
import logging

from models import get_model


def test(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_correct = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            step_loss = loss.item()

            test_loss += step_loss 
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()
            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()

        acc = 100.*float(correct)/float(total)
        
        
    return (test_loss/batch_idx, acc)

def train(net:nn.Module, dataloader, optimizer, criterion, device, max_batchsize):
    net = net.to(device)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with MemoryManagerProxy(is_private=args.private,data_loader=dataloader, max_physical_batch_size=max_batchsize, optimizer=optimizer) as new_dataloader:
        for batch_idx, (data, label) in enumerate(new_dataloader):   
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, label)
            loss.backward()

            optimizer.step()
            step_loss = loss.item()
            
            train_loss += step_loss
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).float().cpu().sum()
            acc = 100.*float(correct)/float(total)
    return (train_loss/batch_idx, acc)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tools.set_rng_seed(args.seed)

    print('==> Preparing data..')
    data_factory = DataFactory(which=args.dataset, data_root=args.data_root)

    trainset = data_factory.getTrainSet()
    testset = data_factory.getTestSet()
    
    trainloader = DataLoader(trainset, batch_size=args.batchsize) 
    testloader = DataLoader(testset, batch_size=args.batchsize)


    print('\n==> Creating model instance')
    net = get_model(args.net, args.dataset, act_func=args.actv).to(device)
    net = ModuleValidator().fix_and_validate(net)
    optimizer = optim.SGD(
            net.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay)

    loss_func = nn.CrossEntropyLoss()
    if(args.private):
        if(args.extra == 'clip_only'):
            args.eps = None
            pe = ClipOnlyPrivacyEngine()
        else:
            pe = opacus.PrivacyEngine()
        net, optimizer, trainloader = pe.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            target_epsilon=args.eps,
            target_delta=args.delta,
            max_grad_norm=args.clip,
            epochs=args.epoch,
        )
    
    print('\n==> Strat training')
    csv_list = []
    for epoch in range(args.epoch):
        print('\nEpoch: %d' % epoch)
        t0 = time.time()
        train_loss, train_acc = train(net,trainloader,optimizer,loss_func,device, MAX_BATCHSIZE.get(args.net,args.batchsize))
        t1 = time.time()
        test_loss, test_acc = test(net, testloader, loss_func, device)
        t2 = time.time()
        csv_list.append((epoch, train_loss, train_acc, test_loss, test_acc, t1-t0, t2-t1))
        print(f'Train loss:{train_loss:.5f} train acc:{train_acc} test loss:{test_loss} test acc:{test_acc} time cost:{t2-t0:.2f}s')

    sess = f"{args.net}_{args.dataset}_e{args.epoch}"
    if(args.private):
        if(args.extra == 'clip_only'):
            sess += '_clip_only'
        else:
            sess += f'_eps{args.eps:.2f}'
        
    csv_path = tools.save_csv(sess, csv_list, f'{pwd}/../exp/{args.actv.lower()}')
    exp_checksum = get_md5(csv_path)
    net_path = tools.save_net(sess, net, f'{pwd}/../trained_net/{args.actv.lower()}')
    model_checksum = get_md5(net_path)

    ent = sqlite_proxy.insert_net(func=args.actv, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path, model_checksum=model_checksum, exp_checksum=exp_checksum, extra=args.extra)
    sqlite_proxy.rpc_insert_net(ent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Differentially Private learning with DP-SGD')

    ## general arguments
    parser.add_argument('--net', default='resnet', type=str, help='network for experiment')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--data_root',default=pwd+'/../dataset', type=str, help='directory of dataset stored or loaded')

    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
    parser.add_argument('--batchsize', default=256, type=int, help='batch size')
    parser.add_argument('--epoch', default=60, type=int, help='total number of epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='base learning rate (default=0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')
    parser.add_argument('--actv', default='relu',type=str)

    ## arguments for learning with differential privacy
    parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
    parser.add_argument('--clip', default=4., type=float, help='gradient clipping bound')
    parser.add_argument('--sigma', default=None, type=float)
    parser.add_argument('--eps', default=None, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    parser.add_argument('--extra', default=None, type=str, choices=['clip_only'], help='clip gradients but no noise addition. Active when args.private is set')

    args = parser.parse_args()

    if(args.private):
        assert (args.eps!=None) ^ (args.extra=='clip_only'),  f'when args.private is set, one of args.eps and clip_only must be set'

    # set max physical batchsize avoiding out of memory  
    MAX_BATCHSIZE={
        'vgg':4,
        'resnet1202':40,
        'resnet8102':4
    }

    main(args)
