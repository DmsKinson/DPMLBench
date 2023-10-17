import sys
import os
import torchvision.transforms as transforms
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import DataFactory

import argparse

import torch
from opacus import PrivacyEngine

from train_utils import  train, test
from data import  get_scatter_transform, get_scattered_loader
from models import  get_model
from models.ScatterLinear import ScatterLinear
import tools
import sqlite_proxy
import time
from DataFactory import TRANSFORM_DICT

FUNC_NAME = 'handcraft'

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tools.set_rng_seed(args.seed)
    pwd = sys.path[0]
    data_factory = DataFactory.DataFactory(args.dataset)
    ori_trans_list = TRANSFORM_DICT[args.dataset]
    ori_trans_list.append(transforms.Resize(size=[32*4,32*4]))
    trainset = data_factory.getTrainSet(transform_list=ori_trans_list)
    testset = data_factory.getTestSet(transform_list=ori_trans_list)
    # initialize scattering net
    scattering, K, (h, w) = get_scatter_transform(trainset[0][0].shape)
    scattering.to(device)

    # initialize original dataloader
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batchsize, shuffle=False)

    if(args.net == 'linear'):
        model = ScatterLinear(K, (h, w), num_groups=args.num_groups)
    else:
        model = get_model(args.net, args.dataset, in_channel=K)

    model.to(device)
    # if there is no data augmentation, pre-compute the scattering transform
    # train_loader = get_scattered_loader(train_loader, scattering, K, w, h, device, 
    #                                     drop_last=True)
    # test_loader = get_scattered_loader(test_loader, scattering, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,)
    criterion = torch.nn.CrossEntropyLoss()
    if(args.private):
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.epoch,
            target_epsilon=args.eps,
            target_delta=args.delta,
            max_grad_norm=args.clip,
        )


    csv_list = []
    sess = f'{args.net}_{args.dataset}_e{args.epoch}'
    if(args.private):
        sess = sess+f'_eps{args.eps}'

    for epoch in range(args.epoch):
        print(f"\nEpoch: {epoch}")
        t0 = time.time()
        train_loss, train_acc = train(model=model,train_loader=train_loader,criterion=criterion,scattering=scattering,optimizer=optimizer,K=K,w=w,h=h,max_physical_bs=args.max_physical_batch_size,private=args.private)
        t1 = time.time()
        test_loss, test_acc = test(model=model,test_loader=test_loader,criterion=criterion,scattering=scattering,K=K,w=w,h=h)
        t2 = time.time()
        csv_list.append((epoch,train_loss,train_acc,test_loss,test_acc, t1-t0, t2-t1))

    csv_path = tools.save_csv(sess, csv_list,f'{pwd}/../exp/{FUNC_NAME}')
    net_path = tools.save_net(sess, model, f'{pwd}/../trained_net/{FUNC_NAME}')

    ent = sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', type=str, default='simple')
    parser.add_argument('--dataset', choices=['cifar10', 'fmnist', 'mnist','svhn'], default='cifar10')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=2)

    parser.add_argument('--private','-p',action='store_true')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--noise_multiplier', type=float, default=1)
    parser.add_argument('--eps',type=float,default=None)
    parser.add_argument('--clip', type=float, default=4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--delta',type=float,default=1e-5)
    parser.add_argument('--input_norm', default='GroupNorm',)
    parser.add_argument('--num_groups', type=int, default=81)
    parser.add_argument('--max_physical_batch_size',type=int,default=8)
    args = parser.parse_args()
    main(args)
