import sys,os

import numpy as np

pwd = sys.path[0]
sys.path.append(pwd+'/..')

import argparse
import torch
from torch.utils.data.dataset import Subset,random_split,Dataset

from torch.utils.data import DataLoader
import torch.optim as optim
from models import get_model
from DataFactory import DataFactory
import tools
from tqdm import tqdm
from data_manager import get_md5
from semisupervise import uda   
from semisupervise.uda_transform import rand_aug_process
import sqlite_proxy
from torchvision import transforms


from tools import test, DatasetWithNewlable
import time

from utils import FeatureExtractor,cal_ensemble_acc,search_sigma,get_epsilon

FUNC_NAME = 'knn'

def aggregation(teacher_preds, sigma):
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

def prepare_stu_dataset(public_set,private_set, is_hog:bool, sample_prob:float, n_teacher:int, sigma:float,root_dir:str, device, model=None):
    """
    Generate noisy dataset for student training by aggregation

    :param public_set: whole public dataset for aggregation
    :param private_set: whole private dataset 
    :param is_hog: whether using hog extractor or model extractor
    :param sample_prob: sample probability for choosing 'teacher' in private dataset
    :param n_teacher: num of teachers
    :param sigma: noise scale for aggregation
    :param device: device
    :param model: model extractor
    """
    print('Prepare student dataset')
    sess = f'n_teacher{args.n_teacher}_n_whole{args.n_whole_samples}'
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
    
    noisy_label = aggregation(teachers_preds, sigma)

    acc = cal_ensemble_acc(public_set, noisy_label,device)
    print('\nNoisy dataset accuracy=',acc)

    noisy_set = DatasetWithNewlable(public_set,noisy_label)

    return noisy_set

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tools.set_rng_seed(args.seed)

    model = get_model(args.net, args.dataset).to(device)

    df = DataFactory(args.dataset)
    private_set = df.getTrainSet()
    whole_set = df.getTestSet('full',transform_list=[transforms.ToTensor(),transforms.Resize(32)])

    feature_root = os.path.join(pwd,'feature',args.dataset,args.net)
    if(not os.path.exists(feature_root)):
        os.makedirs(feature_root)
    indices_path = os.path.join(feature_root,f'indices_{args.n_whole_samples}.pt')
    if(os.path.exists(indices_path)):
        print('Load indices from:',indices_path)
        train_indices,test_indices = torch.load(indices_path)
    else:
        indices = torch.randperm(len(whole_set))
        train_indices,test_indices = indices[:args.n_whole_samples], indices[args.n_whole_samples:]
        torch.save((train_indices,test_indices),indices_path)

    stu_trainset = Subset(whole_set, train_indices)
    stu_testset = Subset(whole_set, test_indices)
    print('\nlen(private_set)=',len(private_set),'len(stu_trainset)=',len(stu_trainset),'len(stu_testset)=',len(stu_testset))
    
    testloader = DataLoader(stu_testset, args.batchsize)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sigma = args.sigma
    if(args.eps == None):
        sigma = 0
        eps = None
    else:
        if(args.sigma is None):
            sigma,eps = search_sigma(args.eps, args.delta, args.sample_prob, args.n_query*args.iter)
        else:
            sigma = args.sigma
            eps = get_epsilon(sigma, args.delta, args.sample_prob, args.n_query*args.iter)
    print(f'sigma : {sigma}',f'eps : {eps}')


    csv_list = []
    for s in range(args.iter):
        print('Stage:',s)
        # use hog in first iteration only
        stu_dataset = prepare_stu_dataset(stu_trainset, private_set, is_hog=(s==0), sample_prob=args.sample_prob, n_teacher=args.n_teacher, sigma=sigma, root_dir=feature_root,device=device,model=model)

        label_size = args.n_query
        indices = torch.randperm(len(stu_dataset))[:label_size]

        label_set = Subset(stu_dataset, indices)
        unlabel_set = stu_dataset
        unlabel_augset = rand_aug_process(unlabel_set)
        sample_rate = args.batchsize/args.n_whole_samples

        # reform student training data
        label_loader = DataLoader(label_set, batch_size=int(len(label_set)*sample_rate))
        unlabel_loader = DataLoader(unlabel_set, batch_size=int(len(unlabel_set)*sample_rate))
        unlabel_aug_loader = DataLoader(unlabel_augset, batch_size=int(len(unlabel_augset)*sample_rate))

        # train student
        for e in range(args.epoch):
            print('Epoch:',e)
            time_cost = time.time()
            label_loss, unlabel_loss = uda.train(model, criterion, optimizer, label_loader,unlabel_loader,unlabel_aug_loader,args.lambda_u, args.mask_threshold, args.temperature_T,device)
            test_loss, test_acc = test(model, testloader, criterion, device)
            time_cost = time.time() - time_cost
            csv_list.append((s,e,label_loss,unlabel_loss,test_loss,test_acc,time_cost))
            print('Test loss:',test_loss, ' Test acc:',test_acc,' Time cost:',time_cost)

    sess = f'{args.net}_{args.dataset}_e{args.epoch}_{args.eps}_uda'
    csv_path = tools.save_csv(sess, csv_list, f'{pwd}/../exp/{FUNC_NAME}')
    exp_checksum = get_md5(csv_path)
    net_path = tools.save_net(sess, model, f'{pwd}/../trained_net/{FUNC_NAME}')
    model_checksum = get_md5(net_path)

    ent = sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path, model_checksum=model_checksum, exp_checksum=exp_checksum, extra='uda')
    sqlite_proxy.rpc_insert_net(ent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='simplenn', type=str, help='network for experiment')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--n_query', default=700, type=int, help='number of set for student training in a supervisored manner')
    parser.add_argument('--n_whole_samples', default=9000, type=int, help='number of samples that student access to')
    parser.add_argument('--data_root', default=pwd+'/../dataset')
    # parser.add_argument('--teacher_root', default='teachers', type=str)
    parser.add_argument('--batchsize', default=512, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--n_teacher', default=800, type=int)

    parser.add_argument('--lambda_u', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--mask_threshold', default=0.1, type=float, help='pseudo label threshold')
    parser.add_argument('--temperature_T', default=0.4, type=float, help='pseudo label temperature')

    parser.add_argument('--eps', default=None, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
    parser.add_argument('--sigma', default=None, type=int, help='')
    parser.add_argument('--iter', default=2, type=int, help='iteration times')
    parser.add_argument('--sample_prob', default=0.15, type=float, help='sample probability from private dataset, gamma in paper')

    args = parser.parse_args()

    main(args)