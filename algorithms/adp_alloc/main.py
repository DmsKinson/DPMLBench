import sys
import os
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(f'{pwd}/../..')

import opacus

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from DataFactory import DataFactory
import noise_scheduler

import argparse
import time

from models import get_model
from tools import MemoryManagerProxy
import tools
import helper
import sqlite_proxy

FUNC_NAME = 'adp_alloc'

def test(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
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
            correct += correct_idx.sum()

        acc = 100.*float(correct)/float(total)
    
    return (test_loss/(batch_idx+1), acc)

def train(net:nn.Module, dataloader, optimizer, criterion, device,):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()

    with MemoryManagerProxy(is_private=True, data_loader=dataloader, max_physical_batch_size=args.phy_bs, optimizer=optimizer) as new_dataloader:
        for batch_idx, (data, label) in enumerate(new_dataloader):   
            data, label = data.to(device), label.to(device)
            outputs = net(data)
            loss = criterion(outputs, label)
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
            step_loss = loss.item()
            
            train_loss += step_loss
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).float().cpu().sum()
            acc = 100.*float(correct)/float(total)
    

    t1 = time.time()
    return (train_loss/(batch_idx+1), acc)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tools.set_rng_seed(args.seed)

    print('==> Preparing data..')
    data_factory = DataFactory(which=args.dataset, data_root=args.data_root)

    trainset = data_factory.getTrainSet();
    testset = data_factory.getTestSet();
    trainloader = DataLoader(trainset, batch_size=args.batchsize)
    testloader = DataLoader(testset, batch_size=args.batchsize)
    
    # calculate sigma with fixed epoch
    if(args.early_stop):
        print('In early stop mode.')
        assert args.sigma != None, 'In early stop mode, sigma must be provided'
        print(f'\n==> Noise multiplier is {args.sigma}')
    else:
        print('In fixed epsilon mode.')
        q = args.batchsize/len(trainset)
        sigma,eps = helper.search_sigma(args.eps, args.epoch, noise_scheduler.get_lambda(args.decay), args.delta, q)
        assert eps > 0, "can't computing proper noise_multiplier" 
        print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(eps, args.delta))
        print(f'\n==> Noise multiplier is {sigma}')


    print('\n==> Creating model instance')
    net = get_model(args.net,args.dataset).to(device)
    optimizer = optim.SGD(
            net.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay)

    loss_func = nn.CrossEntropyLoss()

    
    privacy_engine = opacus.PrivacyEngine()
    net, optimizer, trainloader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        max_grad_norm=args.clip,
        noise_multiplier=args.sigma if args.early_stop else sigma
    )
    
    
    print('\n==> Strat training')
    csv_list = []
    noise_schd = noise_scheduler.ExpDecay(optimizer)
    # noise_schd = ExponentialNoise(optimizer,gamma=0.9)
    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch}')
        t0 = time.time()
        train_loss, train_acc = train(net=net,dataloader=trainloader,optimizer=optimizer,criterion=loss_func,device=device)
        t1 = time.time()
        test_loss, test_acc = test(net, testloader, loss_func, device)
        t2 = time.time()
        total_eps = privacy_engine.get_epsilon(args.delta)
        noise = optimizer.noise_multiplier
        noise_schd.step()
        csv_list.append((epoch, train_loss, train_acc, test_loss, test_acc,noise,total_eps, t1-t0, t2-t1))
        print(f'Train loss:{train_loss:.5f} train acc:{train_acc} test loss:{test_loss} test acc:{test_acc} time cost:{t2-t0:.2f}s')
        print('total_eps=',total_eps)
        #early stop
        if(total_eps >= args.eps):
            break

    sess = f"{args.net}_{args.dataset}_e{args.epoch}_eps{args.eps:.2f}"
    csv_path = tools.save_csv(sess, csv_list, os.path.join(pwd,'..','..','exp',FUNC_NAME))
    net_path = tools.save_net(sess, net, os.path.join(pwd, '..', '..', 'trained_net', FUNC_NAME))

    ent = sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Differentially Private learning with DP-SGD')

    ## general arguments
    parser.add_argument('--net', default='simple', type=str, help='network for experiment')
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
    parser.add_argument('--data_root',default=os.path.join(pwd,'..','..','dataset'), type=str, help='directory of dataset stored or loaded')
    parser.add_argument('--seed', default=5, type=int, help='random seed')
    parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
    parser.add_argument('--batchsize', default=512, type=int, help='batch size')
    parser.add_argument('--epoch', default=60, type=int, help='total number of epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='base learning rate (default=0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')

    ## arguments for learning with differential privacy
    parser.add_argument('--clip', default=4., type=float, help='gradient clipping bound')
    parser.add_argument('--eps', default=1., type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    parser.add_argument('--decay', default='exp', type=str, choices=['time','exp','step','poly'])
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--sigma', default=None, type=float)
    parser.add_argument('--phy_bs', default=8, type=int, help='max physical batch size')

    args = parser.parse_args()
    main(args)
