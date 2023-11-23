import sys
import os
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd, '..', '..'))

import torch
from torch import nn
from torch.utils.data.dataset import  Subset
from DataFactory import DataFactory
from torch.utils.data import DataLoader
from rrprior import rr_prior
import utils
import argparse
from models import get_model
import time
from tqdm import tqdm
import tools
import sqlite_proxy
import copy

# n_label of different dataset 
K_TABLE = {
    'mnist' : 10,
    'cifar10' : 10,
    'fmnist':10,
    'svhn':10,
    'cifar100' : 100
}

def partition(dataset, T) -> list:
    data_list = []
    n_len = len(dataset)
    step = n_len // T
    for t in range(T):
        subset = Subset(dataset, list(range(step*t,step*(t+1))) )
        data_list.append(subset)
    return data_list

def get_K(dataset:str):
    return K_TABLE[dataset.lower()]
    

def test(net, criterion, testloader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_correct = []
    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(testloader):
            inputs, label = inputs.to(device), label.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, label)
            step_loss = loss.item()

            test_loss += step_loss 
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct_idx = predicted.eq(label.data).cpu()
            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()

        acc = 100.*float(correct)/float(total)
    return (test_loss/batch_idx, acc)

def train(net:nn.Module, optimizer, criterion, noise_set, device, epoch):    
    train_loss = 0
    correct = 0
    total = 0
    net.train()
    optimizer.zero_grad()
    try:
        for p in net.parameters():
            del p.grad_batch
    except:
        pass

    for n_batch, (data, noisy_label) in enumerate(noise_set):
        data, noisy_label_p, lam = utils.mixup(data, noisy_label, alpha=args.alpha)
        # data, noisy_label, noisy_label_p = map(Variable, data, noisy_label, noisy_label_p)
        data, noisy_label, noisy_label_p = data.to(device), noisy_label.to(device), noisy_label_p.to(device)
        optimizer.zero_grad()

        output = net(data)
        loss = utils.mixup_criterion(criterion ,output, noisy_label, noisy_label_p, lam)
        loss.backward()
        optimizer.step()

        step_loss = loss.item()
        train_loss += step_loss
        _, predicted = torch.max(output.data, dim=1)
        total += noisy_label.size(0)
        correct += predicted.eq(noisy_label.data).sum().item()
        acc = 100.*float(correct)/float(total)
    return (train_loss/(n_batch+1), acc)

def main(args):

    # set seed
    tools.set_rng_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('==> Preparing data..')
    
    ## preparing data for training && testing
    data_factory = DataFactory(which=args.dataset.lower(), data_root=args.data_root)
    
    # number of label types 
    K = get_K(args.dataset)
    best_acc = 0

    train_set = data_factory.getTrainSet()
    test_set = data_factory.getTestSet()
    testloader = DataLoader(test_set, batch_size=args.batchsize)

    train_set_list = partition(train_set, args.stage)
    noise_set = []

    csv_list = []
    # for every S(t)
    n_acc = 0
    n_label = 0
    for idx, subset in enumerate(train_set_list):
        print("Stage ",idx)

        print('==> Preparing model')
        
        net = get_model(name=args.net,dataset=args.dataset).to(device)
        optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=args.momentum)
        dataloader = DataLoader(subset, args.batchsize, drop_last=True)
        criterion = nn.CrossEntropyLoss()
    

        print('==> Preparing noise data..')
        # generate new dataset with label formed by RRWithPrior
        l_avgk = []
        for data,label in tqdm(dataloader) :
            data, label = data.to(device), label.to(device)
            # pr = (p1,...,pk) be the probabilities predicted by M(t) on xi
            if idx == 0:
                # M(0) ouputs equal probabilies for all classes
                pr = torch.ones([args.batchsize,K]) / K
            else:
                pr = torch.softmax(last_net(data),dim=1)

            avg_k, noise_label = rr_prior(pr, label, K, args.eps)
            l_avgk.append(avg_k)
            noise_label = noise_label.to(device)
            n_acc += (noise_label == label).sum().item()
            n_label += label.shape[0]
            noise_set.append((data,noise_label))
        print("avg_k:",torch.Tensor(l_avgk).mean().item())

        print(f"noisy dataset with {n_label} data has {n_acc/n_label} acc")
        for e in range(args.epoch):
            print(f'\nEpoch {e}:')
            t0 = time.time()
            train_loss, train_acc = train(net, optimizer=optimizer, criterion=criterion, noise_set=noise_set, device=device, epoch=e)
            t1 = time.time()
            test_loss, test_acc = test(net, criterion=criterion, testloader=testloader, device=device)
            t2 = time.time()
            print(f'Train loss:{train_loss:.5f} train acc:{train_acc} test loss:{test_loss} test acc:{test_acc} time cost:{t2-t0:.2f}s')
            if(args.stage == idx + 1):
                csv_list.append((idx, e, train_loss, train_acc, test_loss, test_acc,  t1-t0, t2-t1))

        # cache net
        last_net = copy.deepcopy(net)

    # save train result in csv file
    # stage, epoch, train_loss, train_acc, test_loss, test_acc, best_acc
    FUNC_NAME = f'lp-{args.stage}st'
    sess = f'{args.net}_{args.dataset}_e{args.epoch}_eps{args.eps:.1f}'
    csv_path = tools.save_csv(sess, csv_list, os.path.join(pwd,'..','..','exp',FUNC_NAME))
    net_path = tools.save_net(sess, net, os.path.join(pwd, '..', '..', 'trained_net', FUNC_NAME))

    ent = sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='On Deep Learning with Label Differential Privacy')

    ## general arguments
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
    parser.add_argument('--net', default='simple',type=str, help='model structure for experiment')
    parser.add_argument('--data_root', default=os.path.join(pwd,'..','..','dataset'), type=str, help='directory of dataset stored or loaded')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    # parser.add_argument('--sess', default='mnist_2st_e40_eps8', type=str, help='session name')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    # parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
    parser.add_argument('--batchsize', default=256, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int, help='total number of epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='base learning rate (default=0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')
    parser.add_argument("--stage", "-t", default=2, type=int, help='number of stages')
    parser.add_argument("--alpha", default=4, type=float, help='coefficient of mixup (4 ~ 8 recommended)')


    ## arguments for learning with differential privacy
    parser.add_argument('--eps', default=1, type=float, help='privacy parameter epsilon')

    args = parser.parse_args()

    main(args)