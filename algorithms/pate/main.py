import sys
import os
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd, '..', '..'))

import argparse
import torch
import utils
from torch.utils.data import DataLoader
import torch.optim as optim
from models import get_model
from data_factory import DataFactory
from utils import train, test
import tools
import sqlite_proxy

FUNC_NAME = 'pate'

def predict(net, dataloader: DataLoader, device):
    outputs =[]
    net.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            output = net(images)     # softmax
            ps = torch.argmax(output, dim=1)    # label 1xbs
            outputs.append(ps)
    outputs = torch.cat(outputs,0) # 1xN, N:len(dataset)
    return outputs

def main(args):
    if args.device == None or args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    tools.set_rng_seed(args.seed)

    df = DataFactory(args.dataset)
    testset = df.getTestSet('shadow')
    stu_testset = df.getTestSet()
    new_data, new_labels = utils.obtain_new_data(args.net, args.dataset, args.n_teacher, args.eps, args.delta, testset,args.n_stu_trainset,device)

    acc = utils.cal_ensemble_acc(new_data, new_labels)
    print("accuracy of noisy dataset:",acc)
    
    # replace with noisy dataset
    stu_trainset = utils.DatasetWithNewlable(new_data,new_labels)

    # reform student training data
    stu_trainloader = DataLoader(stu_trainset, batch_size=args.batchsize)
    stu_testloader = DataLoader(stu_testset,batch_size=args.batchsize)
    criterion = torch.nn.CrossEntropyLoss()

    net = get_model(args.net,args.dataset).to(device)
    optimizer = optim.SGD(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    csv_list = []
    for epoch in range(args.epoch):
        print('\nEpoch: %d' % epoch)
        train_loss, train_acc = train(net, stu_trainloader, optimizer=optimizer, criterion=criterion, device=device, epoch=epoch)
        test_loss, test_acc = test(net, stu_testloader, criterion, device=device)
        print('Train loss:%.5f'%(train_loss), 'train acc:', train_acc,'test loss:%.5f'%(test_loss), 'test acc:', test_acc)

        csv_list.append((epoch, train_loss, train_acc, test_loss, test_acc))

    sess = f'{args.net}_{args.dataset}_e{args.epoch}'
    if(args.eps != None):
        sess += f'_eps{args.eps}'
    csv_path = tools.save_csv(sess, csv_list, os.path.join(pwd,'..','..','exp',FUNC_NAME))
    net_path = tools.save_net(sess, net, os.path.join(pwd, '..', '..', 'trained_net', FUNC_NAME))

    ent = sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--net', default='simple', type=str, help='network for experiment')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--n_stu_trainset', default=1000, type=int, help='number of train set for student training aka T in paper')
    parser.add_argument('--n_stu_testset', default=-1, type=int, help='number of test set for student testing (-1 means the rest part of testset removing stu_trainset)')
    parser.add_argument('--data_root', default=os.path.join(pwd,'..','..','dataset'), type=str, help='directory of dataset stored or loaded')

    # parser.add_argument('--teacher_root', default='teachers', type=str)
    parser.add_argument('--batchsize', default=256, type=int)
    parser.add_argument('--epoch', default=60, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--n_teacher', default=100, type=int)

    parser.add_argument('--eps', default=None, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    parser.add_argument('--agg_mode', default='LNMax', type=str, help='LNMax or GNMax')
    parser.add_argument('--sigma1', default=150, type=int, help='')
    parser.add_argument('--sigma2', default=40, type=int, help='')
    # parser.add_argument('--gamma', default=0.025, type=float, help='Laplacian noise of inversed scale')
    parser.add_argument('--threshold', default=50, type=int, help='threshold in GNMax')
    parser.add_argument('--semi', action='store_true')

    args = parser.parse_args()

    main(args)