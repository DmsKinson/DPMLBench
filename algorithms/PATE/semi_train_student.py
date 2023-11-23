import sys,os
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd,'..'))

import argparse
import torch
import utils
from torch.utils.data import DataLoader
import torch.optim as optim
from models import get_model
from DataFactory import DataFactory
from utils import test
import tools
import sqlite_proxy
from semisupervise import generator, improved_GAN

FUNC_NAME = 'pate'

def main(args):
    if args.device == None or args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    tools.set_rng_seed(args.seed)

    df = DataFactory(args.dataset,args.data_root)
    trainset = df.getTestSet('shadow')
    stu_testset = df.getTestSet()
    new_data, new_labels = utils.obtain_new_data(args.net, args.dataset, args.n_teacher, args.eps, args.delta, trainset, args.n_whole_samples, args.n_query, device)

    acc = utils.cal_ensemble_acc(new_data, new_labels)
    print("accuracy of noisy dataset:",acc)
    
    # replace with noisy dataset
    stu_trainset = utils.DatasetWithNewlable(new_data,new_labels)

    # reform student training data
    stu_trainloader = DataLoader(stu_trainset, batch_size=args.batchsize)
    stu_testloader = DataLoader(stu_testset, batch_size=args.batchsize)

    # netD and netG
    net = get_model(args.net,args.dataset).to(device)
    nc = 1 if(args.dataset in ['mnist','fmnist']) else 3
    net_g =  generator.netG(args.nz,args.ngf,nc).to(device)

    optimizerD = optim.Adam(net.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    optimizerG = optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    
    criterionD = torch.nn.CrossEntropyLoss() # binary cross-entropy
    criterionG = torch.nn.MSELoss()
    csv_list = []
    semi_rate = args.n_query / args.n_whole_samples
    for epoch in range(args.epoch):
        print('\nEpoch:',epoch)
        d_loss, g_loss, train_acc = improved_GAN.train(epoch, net, net_g, criterionD, criterionG, optimizerD, optimizerG,stu_trainloader, semi_rate, device)
        test_loss, test_acc = test(net, stu_testloader, criterionD, device)
        print('test_loss:',test_loss, 'test_acc:',test_acc)
        csv_list.append((epoch, d_loss, g_loss, train_acc, test_loss, test_acc))

    sess = f'{args.net}_{args.dataset}_e{args.epoch}'
    if(args.eps != None):
        sess += f'_eps{args.eps}'
    sess += '_semi'

    csv_path = tools.save_csv(sess, csv_list, f'{pwd}/../exp/{FUNC_NAME}')
    net_path = tools.save_net(sess, net, f'{pwd}/../trained_net/{FUNC_NAME}')

    sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path, extra='semi')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--net', default='simplenn', type=str, help='network for experiment')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--data_root', default=os.path.join(pwd,'..','dataset'))
    parser.add_argument('--n_whole_samples', default=1000, type=int, help='number of samples that student access to')
    parser.add_argument('--n_query', type=int, default=100, help='number of samples using the noisy aggregation mechanism, T in paper')
    # parser.add_argument('--teacher_root', default='teachers', type=str)
    parser.add_argument('--batchsize', default=200, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--n_teacher', default=100, type=int)

    parser.add_argument('--eps', default=None, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    # parser.add_argument('--agg_mode', default='LNMax', type=str, help='LNMax or GNMax')
    # parser.add_argument('--sigma1', default=150, type=int, help='')
    # parser.add_argument('--sigma2', default=40, type=int, help='')
    # parser.add_argument('--gamma', default=0.025, type=float, help='Laplacian noise of inversed scale')
    # parser.add_argument('--threshold', default=50, type=int, help='threshold in GNMax')
    parser.add_argument('--nz', default=100, type=int, help='Size of z latent vector')
    parser.add_argument('--ngf', default=32, type=int, help='Number of G output filters')
    parser.add_argument('--momentum', default=0.5, type=float)

    args = parser.parse_args()

    main(args)