import sys,os

import numpy as np
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd, '..', '..'))

import argparse
import torch
from torch.utils.data.dataset import Subset,random_split,Dataset

from torch.utils.data import DataLoader
import torch.optim as optim
from models import get_model
from data_factory import DataFactory
import tools
from tqdm import tqdm
import sqlite_proxy

from tools import train, test, DatasetWithNewlable
import time

from utils import FeatureExtractor,cal_ensemble_acc,search_sigma,get_epsilon

FUNC_NAME = 'knn'

def aggreagation(teacher_preds, sigma):
    """
    :param teacher_preds: obtained label from knn, shape is  n_stu_trainset*n_teacher(k)
    :param sigma : Gaussian noise scale 
    """
    print('gaussian scale=',sigma)
    print('labels shape',teacher_preds.shape)
    noisy_label = torch.zeros(size=(teacher_preds.shape[0],),dtype=torch.long)
    if(sigma != 0):
        gauss = torch.distributions.Normal(0, sigma)

    for i in range(noisy_label.shape[0]):
        label_count = torch.bincount(teacher_preds[i, :], minlength=10).float()
        if(sigma != 0):
            label_count += gauss.sample(sample_shape = label_count.shape)
        noisy_label[i] = torch.argmax(label_count)

    return noisy_label

def prepare_stu_dataset(public_set,private_set, is_hog:bool, sample_prob:float, n_teacher:int, sigma:float, root_dir:str, device, model=None):
    print('Prepare student dataset')
    sess = f'{args.net}_{args.dataset}_{args.n_teacher}_{args.n_stu_trainset}'
    fe = FeatureExtractor(root_dir, sess, device)
    if(is_hog):
        pri_feature, pri_label = fe.extract_feature_with_hog(private_set, prefix='pri')
        pub_feature, pub_label = fe.extract_feature_with_hog(public_set, prefix='pub')
    else:
        bs = 256
        pri_loader = DataLoader(private_set, bs)
        pub_loader = DataLoader(public_set,bs)
        pri_feature, pri_label = fe.extract_feature_with_model(model, pri_loader, prefix='pri')
        pub_feature, pub_label = fe.extract_feature_with_model(model, pub_loader, prefix='pub')
    pri_feature, pub_feature = pri_feature.cpu().detach(), pub_feature.cpu().detach()
    pri_label, pub_label = pri_label.cpu().detach(), pub_label.cpu().detach()

    num_pri = pri_feature.shape[0]
    teachers_preds = []

    print('\nComputing K-NN')
    for query in tqdm(pub_feature):
        select_teacher = np.random.choice(num_pri, size=int(sample_prob*num_pri))
        dis = np.linalg.norm(pri_feature[select_teacher] - query, axis=1)
        k_idx = select_teacher[np.argsort(dis)[:n_teacher]]
        teachers_preds.append(pri_label[k_idx])
    teachers_preds = torch.stack(teachers_preds) # n_stu_trainset*n_teacher
    
    noisy_label = aggreagation(teachers_preds, sigma)

    acc = cal_ensemble_acc(public_set, noisy_label, device)
    print('\nNoisy dataset accuracy=',acc)

    noisy_set = DatasetWithNewlable(public_set,noisy_label)

    return noisy_set

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tools.set_rng_seed(args.seed)

    model = get_model(args.net, args.dataset).to(device)

    df = DataFactory(args.dataset)
    feature_root = os.path.join(pwd,'feature',args.dataset,args.net)

    if(not os.path.isdir(feature_root)):
        os.makedirs(feature_root)
    private_set = df.getTrainSet()
    testset = df.getTestSet()
    public_set, testset = random_split(testset, [args.n_stu_trainset, len(testset) - args.n_stu_trainset])
    print('\nlen(private_set)=',len(private_set),'len(public_set)=',len(public_set),'len(test_set)=',len(testset))

    testloader = DataLoader(testset, args.batchsize)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sigma = args.sigma
    if(args.eps == None):
        sigma = 0
        eps = None
    else:
        if(args.sigma is None):
            sigma,eps = search_sigma(args.eps, args.delta, args.sample_prob, args.n_stu_trainset)
        else:
            sigma = args.sigma
            eps = get_epsilon(sigma, args.delta, args.sample_prob, args.n_stu_trainset)
    print(f'sigma : {sigma}',f'eps : {eps}')


    csv_list = []
    for s in range(args.iter):
        print('Stage:',s)
        stu_dataset = prepare_stu_dataset(public_set, private_set, is_hog=(s==0), sample_prob=args.sample_prob, n_teacher=args.n_teacher, sigma=sigma, root_dir=feature_root, device=device,model=model)
        trainloader = DataLoader(stu_dataset, args.batchsize)
        # train student
        for e in range(args.epoch):
            print('Epoch:',e)
            time_cost = time.time()
            train_loss, train_acc = train(model, trainloader, optimizer, criterion, device)
            test_loss, test_acc = test(model, testloader, criterion, device)
            time_cost = time.time() - time_cost
            csv_list.append((s,e,train_loss,train_acc,test_loss,test_acc,time_cost))
            print('Train loss:',train_loss, ' Train acc:',train_acc, ' Test loss:',test_loss, ' Test acc:',test_acc,' Time cost:',time_cost)

    sess = f'{args.net}_{args.dataset}_e{args.epoch}_{args.eps}'
    csv_path = tools.save_csv(sess, csv_list, os.path.join(pwd,'..','..','exp',FUNC_NAME))
    net_path = tools.save_net(sess, model, os.path.join(pwd, '..', '..', 'trained_net', FUNC_NAME))

    ent = sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='simple', type=str, help='network for experiment')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--n_stu_testset', default=1000, type=int, help='number of test set for student testing')
    parser.add_argument('--n_stu_trainset', default=1000, type=int, help='number of train set for student training')
    parser.add_argument('--data_root', default=os.path.join(pwd,'..','..','dataset'), type=str, help='directory of dataset stored or loaded')

    parser.add_argument('--batchsize', default=512, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--n_teacher', default=800, type=int)

    parser.add_argument('--eps', default=None, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    parser.add_argument('--sigma', default=None, type=int, help='')
    parser.add_argument('--iter', default=2, type=int, help='iteration times')
    parser.add_argument('--sample_prob', default=0.15, type=float, help='sample probability from private dataset')
    args = parser.parse_args()

    main(args)