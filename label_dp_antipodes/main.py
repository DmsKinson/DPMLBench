#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Runs CIFAR10 and CIFAR100 training with ALIBI for Label Differential Privacy
"""

import pathlib
import sys
import os
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..")

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed
from DataFactory import DataFactory
import tools
import time
# import torch.utils.tensorboard as tensorboard
import torchvision.transforms as transforms

from libs import models
from libs.alibi import Ohm, RandomizedLabelPrivacy, NoisedDataset
from models import get_model
# from opacus.utils import stats
from tqdm import tqdm
import sqlite_proxy
from data_manager import get_md5

FUNC_NAME = 'alibi'

def accuracy(preds, labels):
    return (preds == labels).mean() * 100

def train(model, train_loader, optimizer, criterion:Ohm, device):
    model.train()
    losses = []
    acc = []
    for i, batch in enumerate(train_loader):

        images = batch[0].to(device)
        targets = batch[1].to(device)
        labels = targets if len(batch) == 2 else batch[2].to(device)

        # compute output
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = labels.detach().cpu().numpy()

        # measure accuracy and record loss
        acc1 = accuracy(preds, labels)

        losses.append(loss.item())
        acc.append(acc1)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

    train_loss = np.mean(losses)
    train_acc = np.mean(acc)
    return train_loss, train_acc


def test(model, test_loader, criterion, device):
    model.eval()
    losses = []
    acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            acc.append(acc1)

    test_loss = np.mean(losses)
    test_acc = np.mean(acc)
    return test_loss, test_acc


def adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 30:  # warm-up
        lr = lr * float(epoch + 1) / 30
    else:
        lr = lr * (0.2 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def create_model(arch: str, num_classes: int):
    if "wide" in arch.lower():
        print("Created Wide Resnet Model!")
        return models.wideresnet(
            depth=28,
            widen_factor=8 if num_classes == 100 else 4,
            dropout=0,
            num_classes=num_classes,
        )
    else:
        print("Created simple Resnet Model!")
        return models.resnet18(num_classes=num_classes)


#######################################################################
# main worker
#######################################################################


def main(args):
    tools.set_rng_seed(args.seed)

    best_acc = 0
    num_classes = 10
    model = get_model(args.net, args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # DEFINE LOSS FUNCTION (CRITERION)
    # standard ALIBI in paper
    noise_only_once = True
    randomized_label_privacy = RandomizedLabelPrivacy(
        sigma=args.sigma,
        delta=args.delta,
        mechanism=args.mechanism,
        eps=args.eps,
        device=device
    )
    criterion = Ohm(
        privacy_engine=randomized_label_privacy,
        post_process=args.post_process,
    )
    # DEFINE OPTIMIZER
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # train data
    print('preparing noise dataset')
    df = DataFactory(which=args.dataset, data_root=args.data_root)
    trainset = df.getTrainSet()

    if noise_only_once:
        train_dataset = NoisedDataset(
            args.dataset, trainset, num_classes, randomized_label_privacy
        )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=True,
    )

    # test data
    test_dataset = df.getTestSet()
    test_loader = data.DataLoader(
        test_dataset, batch_size=args.batchsize, shuffle=False
    )

    epsilon, alpha = randomized_label_privacy.privacy
    sigma = randomized_label_privacy.sigma
    label_change = 0
    label_change = train_dataset.label_change
    print(f'sigma={sigma}, eps={epsilon}, alpha={alpha}, label_change={label_change}')

    csv_list = []
    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch}:')
        # train for one epoch
        t0 = time.time()
        randomized_label_privacy.train()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        t1 = time.time()
        # evaluate on validation set
        randomized_label_privacy.eval()
        test_loss, test_acc = test(model, test_loader, criterion, device)
        t2 = time.time()
        csv_list.append((epoch, train_loss, train_acc, test_loss, test_acc, t1-t0, t2-t1))
        print(f'Train loss:{train_loss:.5f} train acc:{train_acc} test loss:{test_loss} test acc:{test_acc} time cost:{t2-t0:.2f}s')

    sess = f"{args.net}_{args.dataset}_e{args.epoch}_eps{args.eps:.2f}"

    csv_path = tools.save_csv(sess, csv_list,f'{pwd}/../exp/{FUNC_NAME}')
    exp_checksum = get_md5(csv_path)
    net_path = tools.save_net(sess, model, f'{pwd}/../trained_net/{FUNC_NAME}')
    model_checksum = get_md5(net_path)

    ent = sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path, model_checksum=model_checksum, exp_checksum=exp_checksum)
    sqlite_proxy.rpc_insert_net(ent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LabelDP Training with ALIBI")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to run training",)
    parser.add_argument("--net", type=str, default="full_inception",)
    parser.add_argument('--data_root', default=pwd+'/../dataset', type=str, help='directory of dataset stored or loaded')

    # learning
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="LR momentum")
    parser.add_argument("--weight_decay", default=0, type=float, help="LR weight decay")
    parser.add_argument("--epoch", default=40, type=int, help="maximum number of epochs",)
    parser.add_argument('--batchsize', default=64, type=int,)

    # Privacy
    parser.add_argument('--delta', type=float, default=1e-5, )
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise multiplier (default 1.0)",)
    parser.add_argument("--eps", type=float, default=8, help="privacy parameter epsilon")
    parser.add_argument("--post_process", type=str, default="mapwithprior", help="Post-processing scheme for noised labels ""(MinMax, SoftMax, MinProjection, MAP, MAPWithPrior, RandomizedResponse)",)
    parser.add_argument("--mechanism", type=str, default="Laplace", help="Noising mechanism (Laplace or Gaussian)",)

    parser.add_argument("--seed", type=int, default=11337, help="Seed")

    args = parser.parse_args()

    main(args)
