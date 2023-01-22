import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import torch
import opacus
from DataFactory import DataFactory
from torch.utils.data.dataloader import DataLoader
from models import get_model
import argparse
import tools
import torch.nn as nn
from loss_func import MSE_Focal, MSE_Focal_L2
import sqlite_proxy
import time
from tools import MemoryManagerProxy
from data_manager import get_md5

FUNC_NAME = 'loss'

def train(net: nn.Module, dataloader, optimizer, criterion, device, epoch, thrs_epoch=0, alpha_emb=1):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    with MemoryManagerProxy(is_private=args.private, data_loader=dataloader, max_physical_batch_size=4, optimizer=optimizer) as new_dataloader:
        for batch_idx, (data, label) in enumerate(new_dataloader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, label, epoch, thrs_epoch, alpha_emb)
            loss.backward()

            optimizer.step()
            step_loss = loss.item()

            train_loss += step_loss
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).float().cpu().sum()
            acc = 100.*float(correct)/float(total)

    return (train_loss/batch_idx, acc)


def test(net, epoch, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_correct = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets, epoch)
            step_loss = loss.item()

            test_loss += step_loss
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()
            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()

        acc = 100.*float(correct)/float(total)
        

    return (test_loss/batch_idx, acc)


def main(args):
    tools.set_rng_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_factory = DataFactory(args.dataset, pwd+'/../dataset')

    trainset = data_factory.getTrainSet()
    testset = data_factory.getTestSet()
    trainloader = DataLoader(
        trainset, batch_size=args.batchsize)
    testloader = DataLoader(testset, batch_size=args.batchsize)

    net = get_model(args.net, args.dataset).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    criterion = MSE_Focal
    if(args.private):
        privacy_engine = opacus.PrivacyEngine()
        net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            target_epsilon=args.eps,
            target_delta=args.delta,
            max_grad_norm=args.clip,
            epochs=args.epoch,
        )
    
    csv_list = []
    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch}:')
        t0 = time.time()
        train_loss, train_acc = train(net=net, dataloader=trainloader, optimizer=optimizer, epoch=epoch, criterion=criterion, device=device)
        t1 = time.time()
        test_loss, test_acc = test(net, epoch, testloader, criterion, device)
        t2 = time.time()
        csv_list.append((epoch, train_loss, train_acc, test_loss, test_acc, t1-t0, t2-t1))
        print(f'Train loss:{train_loss:.5f} train acc:{train_acc} test loss:{test_loss} test acc:{test_acc} time cost:{t2-t0:.2f}s')

    # if(args.private):
    #     eps, alpha = privacy_engine.get_privacy_spent(args.delta)
    #     print(f"eps={eps} , alpha={alpha}")
    sess = f'{args.net}_{args.dataset}_e{args.epoch}'
    if(args.private):
        sess = sess+f'_eps{args.eps}'
    
    csv_path = tools.save_csv(sess, csv_list,f'{pwd}/../exp/{FUNC_NAME}')
    exp_checksum = get_md5(csv_path)
    net_path = tools.save_net(sess, net, f'{pwd}/../trained_net/{FUNC_NAME}')
    model_checksum = get_md5(net_path)

    ent = sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path, model_checksum=model_checksum, exp_checksum=exp_checksum)
    sqlite_proxy.rpc_insert_net(ent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='simple', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epoch', default=60, type=int)
    parser.add_argument('--batchsize', default=512, type=int)
    parser.add_argument('--seed', default=2, type=int)

    parser.add_argument('--private', '-p', action='store_true')
    parser.add_argument('--clip', default=4, type=float)
    parser.add_argument('--eps', default=None, type=float)
    parser.add_argument('--delta', default=1e-5, type=float)
    parser.add_argument('--sigma', default=1.23, type=float)
    args = parser.parse_args()
    main(args)
