import sys
pwd = sys.path[0]
sys.path.append(pwd+'/..')

from pathlib import Path
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from models import get_model
from DataFactory import DataFactory

from torch.optim.sgd import SGD 
import tools
import time

def train(net:nn.Module, dataloader, optimizer, criterion, device, epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, label) in enumerate(dataloader):   
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

    return (train_loss/(batch_idx+1), acc)

def test(net, testloader, criterion, device):
    global best_acc
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
        
    return (test_loss/(batch_idx+1), acc)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_epoch = 0  

    tools.set_rng_seed(args.seed)

    print('==> Preparing data..')
    data_factory = DataFactory(which=args.dataset, data_root=args.data_root)

    trainset = data_factory.getTrainSet()
    testset = data_factory.getTestSet()
    data_size = len(trainset)//args.n_teacher
    trainset = Subset(trainset,list(range(len(trainset)))[args.teacher_id*data_size:(args.teacher_id+1)*data_size])
    print(f'==> Subset has {len(trainset)} data')
    trainloader = DataLoader(trainset, batch_size=args.batchsize) 
    testloader = DataLoader(testset, batch_size=args.batchsize)

    print('\n==> Creating model instance')    
    net = get_model(args.net, args.dataset).to(device)
    optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, )
    loss_func = nn.CrossEntropyLoss()
    
    print('\n==> Strat training')
    for epoch in range(start_epoch, args.epoch):
        print('\nEpoch: %d' % epoch)
        start = time.time()
        train_loss, train_acc = train(net,trainloader,optimizer,loss_func,device, epoch)
        test_loss, test_acc = test(net, testloader, loss_func, device)
        print('Train loss:%.5f'%(train_loss), 'train acc:', train_acc, 'test loss:%.5f'%test_loss, 'test acc:', test_acc)
        print(f'time cost:{time.time()-start:.2f} s',)

    print('\n==> Save teacher model')
    model_dir = Path(pwd).joinpath('..', 'trained_net', 'pate', args.net, args.dataset, f'{args.n_teacher}_teachers')
    model_dir.mkdir(parents=True,exist_ok=True)
    datapath = model_dir.joinpath(f'teacher_stat.csv')
    with open(datapath, 'at') as f:
        f.write(f'{args.teacher_id}, {test_acc}\n')

    sess = f'{args.teacher_id}th_teacher'
    tools.save_net(sess, net, model_dir.as_posix())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Differentially Private learning with PATE')

    ## general arguments
    parser.add_argument('--net', default='simple', type=str, help='network for experiment')
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
    parser.add_argument('--data_root', default=pwd+'/../dataset/', type=str, help='directory of dataset stored or loaded')
    parser.add_argument('--seed', default=2, help= 'random seed')
    parser.add_argument('--teacher_root', default=pwd+'/../trained_net/pate', type=str, help='directory of noisy datset stored or loaded')
    parser.add_argument('--n_teacher', default=250, type=int, help='number of teachers')
    parser.add_argument('--epoch', default=60, type=int, help='total number of epochs')
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--teacher_id', default=0, type=int, help='specific teacher id (start from 0)')

    parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='base learning rate (default=0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')

    args = parser.parse_args()
    main(args)