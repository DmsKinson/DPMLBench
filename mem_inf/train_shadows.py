import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd, '..'))

from pathlib import Path
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from models import get_model
from data_factory import DataFactory
import opacus
from tools import MemoryManagerProxy

from torch.optim.sgd import SGD 
import tools
import time
import sqlite_proxy
from clip_only_utils import ClipOnlyPrivacyEngine


def train(net:nn.Module, dataloader, optimizer, criterion, device, max_phy_bs):

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    with MemoryManagerProxy(is_private=args.private,data_loader=dataloader, max_physical_batch_size=max_phy_bs,optimizer=optimizer) as new_loader:
        for batch_idx, (data, label) in enumerate(new_loader):   
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

    print('Train loss:%.5f'%(train_loss/(batch_idx+1)), 'train acc:', acc, end=' ')
    return (train_loss/(batch_idx+1), acc)

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
        print('test loss:%.5f'%(test_loss/(batch_idx+1)), 'test acc:', acc, end=' ')
        
    return (test_loss/(batch_idx+1), acc)

def main(args):
    MAX_PHY_BS={
        'vgg':8
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tools.set_rng_seed(args.seed)

    print('==> Preparing data...')
    data_factory = DataFactory(which=args.dataset, data_root=args.data_root)
    if(args.extra == 'uda'):
        trainset = data_factory.getUdaTrainset('shadow')
        testset = data_factory.getUdaTestset('shadow')
    else:
        trainset = data_factory.getTrainSet('shadow')
        testset = data_factory.getTestSet('shadow')
    print('len(trainset)=',len(trainset),'len(testset)=',len(testset))
    trainloader = DataLoader(trainset, batch_size=args.batchsize) 
    testloader = DataLoader(testset, batch_size=args.batchsize)

    print('\n==> Creating model instance...')    
    net = get_model(args.net, args.dataset).to(device)
    optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, )
    loss_func = nn.CrossEntropyLoss()

    if(args.private):
        if(args.extra=='clip_only'):
            pe = ClipOnlyPrivacyEngine()
        else:
            pe = opacus.PrivacyEngine()
        net, optimizer, trainloader = pe.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            target_epsilon=args.eps,
            target_delta=args.delta,
            epochs=args.epoch,
            max_grad_norm=args.clip
        )

    print('\n==> Training...')
    csv_list = []
    for epoch in range(args.epoch):
        print('\nEpoch: %d' % epoch)
        start = time.time()
        train_loss, train_acc = train(net, trainloader, optimizer, loss_func, device,max_phy_bs=MAX_PHY_BS.get(args.net,args.batchsize))
        test_loss, test_acc = test(net, testloader, loss_func, device)
        csv_list.append((epoch, train_loss, train_acc, test_loss, test_acc))
        print(f'time cost:{time.time()-start:.2f} s',)

    print('\n==> Saving shadow model...')
    model_dir = Path(pwd).joinpath('..', 'trained_net', 'shadow')
    exp_dir = Path(pwd).joinpath('..','exp','shadow')
    model_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    sess = f'{args.net}_{args.dataset}_e{args.epoch}'
    if(args.private):
        sess += f'_{args.eps}'
    if(args.extra != None):
        sess += f'_{args.extra}'
    csv_path = tools.save_csv(sess, csv_list, exp_dir.as_posix())
    net_path = tools.save_net(sess, net, model_dir.as_posix())

    ent = sqlite_proxy.insert_net(net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), model_type=sqlite_proxy.TYPE_SHADOW, exp_loc=csv_path, model_loc=net_path, extra=args.extra)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## general arguments
    parser.add_argument('--net', default='simple', type=str, help='network for experiment')
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
    parser.add_argument('--data_root', default=pwd+'/../dataset/', type=str, help='directory of dataset stored or loaded')
    parser.add_argument('--seed', default=2, help= 'random seed')
    parser.add_argument('--epoch', default=50, type=int, help='total number of epochs')
    parser.add_argument('--private','-p',action='store_true')
    parser.add_argument('--eps', default=None, type=float)
    parser.add_argument('--delta', default=1e-5, type=float)
    parser.add_argument('--clip', default=4, type=float)
    parser.add_argument('--extra', default=None, type=str, choices=['clip_only','uda',None])

    parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
    parser.add_argument('--batchsize', default=256, type=int, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='base learning rate (default=0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')

    args = parser.parse_args()
    main(args)