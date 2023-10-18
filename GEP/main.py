from math import isnan
import sys
import os

from opacus import GradSampleModule
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+'/../.')

from DataFactory import DataFactory
import tools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse

import time
import sqlite_proxy

import models
from utils import get_sigma, flatten_tensor
from basis_matching import GEP

FUNC_NAME = 'gep'

def group_params(num_p, groups:int):
    assert groups >= 1

    p_per_group = num_p//groups
    num_param_list = [p_per_group] * (groups-1)
    # deal with last group
    num_param_list = num_param_list + [num_p-sum(num_param_list)]
    return num_param_list

def train(gep:GEP, net, trainloader, optimizer, criterion, device, noise0, noise1):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx,(inputs,targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        logging = batch_idx % 20 == 0
        ## compute anchor subspace
        net.zero_grad()
        gep.get_anchor_space(net, loss_func=criterion, logging=logging)
        ## collect batch gradients
        batch_grad_list = []
        net.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        for p in net.parameters():
            batch_grad_list.append(p.grad_sample)  ## bs*...
        ## compute gradient embeddings and residual gradients
        clipped_theta, residual_grad, target_grad = gep(flatten_tensor(batch_grad_list), logging = logging)
        ## add noise to guarantee differential privacy
        theta_noise = torch.normal(0, noise0*args.clip0/args.batchsize, size=clipped_theta.shape, device=clipped_theta.device)
        grad_noise = torch.normal(0, noise1*args.clip1/args.batchsize, size=residual_grad.shape, device=residual_grad.device)
        clipped_theta += theta_noise
        residual_grad += grad_noise
        ## update with Biased-GEP or GEP
        if(args.rgp):
            noisy_grad = gep.get_approx_grad(clipped_theta) + residual_grad
        else:
            noisy_grad = gep.get_approx_grad(clipped_theta)
        if(logging):
            print('target grad norm: %.2f, noisy approximation norm: %.2f'%(target_grad.norm().item(), noisy_grad.norm().item()))
        ## make use of noisy gradients
        offset = 0
        for p in net.parameters():
            shape = p.grad.shape
            numel = p.grad.numel()
            p.grad.data = noisy_grad[offset:offset+numel].view(shape) #+ 0.1*torch.mean(pub_grad, dim=0).view(shape)
            offset+=numel

        optimizer.step()
        step_loss = loss.item()
        
        train_loss += step_loss
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).float().cpu().sum()
        acc = 100.*float(correct)/float(total)
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
            if(args.private):
                step_loss /= inputs.shape[0]

            test_loss += step_loss 
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()
            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()

        acc = 100.*float(correct)/float(total)

    return (test_loss/batch_idx, acc)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_epoch = 0  
    tools.set_rng_seed(args.seed)

    print('==> Preparing data..')
    ## preparing data for training && testing
    datafactory = DataFactory(args.dataset,args.dataroot)
    trainset = datafactory.getTrainSet()
    testset = datafactory.getTestSet()
    trainloader = DataLoader(trainset, batch_size=args.batchsize)
    testloader = DataLoader(testset, batch_size=args.batchsize)

    ## preparing auxiliary data
    n_training = len(trainset)
    n_test = len(testset)
    
    num_public_examples = args.aux_data_size
    # single channel dataset 
    if(args.dataset.lower() in ('mnist','fmnist')):
        public_set = DataFactory(args.dataset,args.dataroot).getPubSet()
        public_loader = DataLoader(public_set,batch_size=num_public_examples)
        public_inputs, _ = next(iter(public_loader))
    else:
        public_inputs = torch.load(pwd+'/'+'imagenet_examples_2000')[:num_public_examples]
    public_targets = torch.randint(high=10, size=(num_public_examples,))
        
    public_inputs, public_targets = public_inputs.to(device), public_targets.to(device)
    print('# of training examples: ', n_training, '# of testing examples: ', n_test, '# of auxiliary examples: ', num_public_examples)


    print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(args.eps, args.delta))
    sample_rate=args.batchsize/n_training
    sigma, eps = get_sigma(sample_rate, args.epoch, args.eps, args.delta, rgp=args.rgp)
    noise_multiplier0 = noise_multiplier1 = sigma
    print('noise scale for gradient embedding: ', noise_multiplier0, 'noise scale for residual gradient: ', noise_multiplier1, '\n rgp enabled: ', args.rgp, ' guarantee: ', eps)
    print('\n==> Creating GEP class instance')

    print(f'\n==> Creating {args.net} model instance')
    net = models.get_model(args.net,args.dataset) 
    net = GradSampleModule(net).to(device)

    loss_func = nn.CrossEntropyLoss(reduction='mean')
    
    num_params = 0
    for p in net.parameters():
        num_params += p.numel()

    print('total number of parameters: ', num_params/(10**6), 'M')

    print('\n==> Dividing parameters in to %d groups'%args.num_groups)
    gep = GEP(args.num_bases, args.batchsize, args.clip0, args.clip1, args.power_iter).to(device)
    gep.num_param_list = group_params(num_params, args.num_groups)
    ## attach auxiliary data to GEP instance
    gep.public_inputs = public_inputs
    gep.public_targets = public_targets

    optimizer = optim.SGD(
            net.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay)

    print('\n==> Strat training')
    csv_list = []
    for epoch in range(start_epoch, args.epoch):
        print(f'\nEpoch {epoch}:')
        t0 = time.time()
        train_loss, train_acc = train(gep, net, trainloader, optimizer, loss_func, device, noise_multiplier0, noise_multiplier1)
        t1 = time.time()
        test_loss, test_acc = test(net, testloader, loss_func, device)
        t2 = time.time()
        csv_list.append((epoch,train_loss, train_acc, test_loss, test_acc,t1-t0,t2-t1))
        print(f'Train loss:{train_loss:.5f} train acc:{train_acc} test loss:{test_loss} test acc:{test_acc} time cost:{t2-t0:.2f}s')
        if(isnan(train_loss)):
            print('Loss become NaN, terminate training process.')


    sess = f'{args.net}_{args.dataset}_e{args.epoch}'
    if(args.private):
        sess += f'_eps{eps:.2f}'

    csv_path = tools.save_csv(sess, csv_list,f'{pwd}/../exp/{FUNC_NAME}')
    net_path = tools.save_net(sess, net, f'{pwd}/../trained_net/{FUNC_NAME}')

    sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Differentially Private learning with GEP')

    ## general arguments
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--net', default='resnet', type=str, help='network for experiment')
    parser.add_argument('--dataroot', default=pwd+'/../dataset', type=str, help='directory of dataset stored or loaded')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--weight_decay', default=2e-4, type=float, help='weight decay')
    parser.add_argument('--batchsize', default=1000, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int, help='total number of epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='base learning rate (default=0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')


    ## arguments for learning with differential privacy
    parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
    parser.add_argument('--eps', default=None, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')

    parser.add_argument('--rgp', action='store_true', help='use residual gradient perturbation or not')
    parser.add_argument('--clip0', default=5., type=float, help='clipping threshold for gradient embedding')
    parser.add_argument('--clip1', default=2., type=float, help='clipping threshold for residual gradients')
    parser.add_argument('--power_iter', default=1, type=int, help='number of power iterations')
    parser.add_argument('--num_groups', default=3, type=int, help='number of parameters groups')
    parser.add_argument('--num_bases', default=1000, type=int, help='dimension of anchor subspace')

    parser.add_argument('--aux_data_size', default=2000, type=int, help='size of the auxiliary dataset')

    args = parser.parse_args()
    main(args)