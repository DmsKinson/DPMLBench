import sys,os
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd,'..','..'))

import argparse
import torch
import utils
from torch.utils.data import DataLoader,Subset
import torch.optim as optim
from models import get_model
from data_factory import DataFactory
from utils import test
import tools
from semisupervise import uda   
from semisupervise.uda_transform import rand_aug_process
from torchvision import transforms
import sqlite_proxy

FUNC_NAME = 'pate'

def main(args):
    if args.device == None or args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    tools.set_rng_seed(args.seed)

    df = DataFactory(args.dataset, args.data_root)
    whole_set = df.getTestSet('full',transform_list=[transforms.ToTensor(),transforms.Resize(32)])

    teacher_root = os.path.join(pwd,'..','..','trained_net','pate',args.net,args.dataset,f'{args.n_teacher}_teachers')
    if(not os.path.isdir(teacher_root)):
        os.makedirs(teacher_root)
    indices_path = os.path.join(teacher_root,f'indices_{args.n_whole_samples}.pt')
    if(os.path.exists(indices_path)):
        print('Load indices from:',indices_path)
        train_indices,test_indices = torch.load(indices_path)
    else:
        indices = torch.randperm(len(whole_set))
        train_indices,test_indices = indices[:args.n_whole_samples], indices[args.n_whole_samples:]
        torch.save((train_indices,test_indices),indices_path)

    stu_trainset = Subset(whole_set, train_indices)
    stu_testset = Subset(whole_set, test_indices)

    new_dataset, new_labels = utils.obtain_new_data(args.net, args.dataset, args.n_teacher, args.eps, args.delta, stu_trainset, args.n_query, device)
    
    print("Calculating ensemble accuracy...")
    acc = utils.cal_ensemble_acc(new_dataset, new_labels, device)
    print("accuracy of noisy dataset:",acc)
    # replace with noisy dataset
    stu_trainset = utils.DatasetWithNewlable(new_dataset,new_labels)

    label_size = args.n_query
    indices = torch.randperm(len(stu_trainset))[:label_size]

    # unlabel training uses the whole set, and label training uses the $label_size random samples 
    label_set = Subset(stu_trainset,indices)
    unlabel_set = stu_trainset

    unlabel_augset = rand_aug_process(unlabel_set)
    sample_rate = args.batchsize/args.n_whole_samples
    
    # reform student training data
    label_loader = DataLoader(label_set, batch_size=int(len(label_set)*sample_rate))
    unlabel_loader = DataLoader(unlabel_set, batch_size=int(len(unlabel_set)*sample_rate))
    unlabel_aug_loader = DataLoader(unlabel_augset, batch_size=int(len(unlabel_augset)*sample_rate))
    stu_testloader = DataLoader(stu_testset, batch_size=args.batchsize)

    net = get_model(args.net,args.dataset).to(device)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    criterion = torch.nn.CrossEntropyLoss() # binary cross-entropy
    csv_list = []

    for epoch in range(args.epoch):
        print('\nEpoch:',epoch)
        label_loss, unlabel_loss = uda.train(net, criterion, optimizer, label_loader,unlabel_loader,unlabel_aug_loader,args.lambda_u, args.mask_threshold, args.temperature_T,device)
        test_loss, test_acc = test(net, stu_testloader, criterion, device)
        print('test_loss:',test_loss, 'test_acc:',test_acc)
        csv_list.append((epoch, label_loss, unlabel_loss, test_loss, test_acc))

    sess = f'{args.net}_{args.dataset}_e{args.epoch}'
    if(args.eps != None):
        sess += f'_eps{args.eps}'
    sess += '_uda'

    csv_path = tools.save_csv(sess, csv_list, os.path.join(pwd,'..','..','exp',FUNC_NAME))
    net_path = tools.save_net(sess, net, os.path.join(pwd, '..', '..', 'trained_net', FUNC_NAME))

    sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path, extra='uda')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--net', default='simplenn', type=str, help='network for experiment')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--data_root', default=os.path.join(pwd,'..','..','dataset'))
    parser.add_argument('--n_whole_samples', default=9000, type=int, help='number of samples that student access to')
    parser.add_argument('--n_query', type=int, default=100, help='number of samples using the noisy aggregation mechanism, T in paper')
    # parser.add_argument('--teacher_root', default='teachers', type=str)
    parser.add_argument('--batchsize', default=200, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--n_teacher', default=100, type=int)
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--lambda_u', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--mask_threshold', default=0.1, type=float, help='pseudo label threshold')
    parser.add_argument('--temperature_T', default=0.4, type=float, help='pseudo label temperature')
    parser.add_argument('--eps', default=None, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    parser.add_argument('--momentum', default=0.5, type=float)

    args = parser.parse_args()

    main(args)