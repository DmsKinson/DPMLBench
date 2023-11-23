import sys,os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd,'..','..'))

import torch
from DataFactory import DataFactory
from models import get_model
import argparse
import tools
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transform
import os
import sqlite_proxy


FUNC_NAME = 'dpgen'

def main(args):
    device = torch.device('cuda' if(torch.cuda.is_available()) else 'cpu')
    tools.set_rng_seed(args.seed)

    dataset_dir = os.path.join(pwd,'label_dataset',args.dataset,f'{args.eps}.pt')
    assert os.path.exists(dataset_dir), f'{dataset_dir} is not existed.'
    
    print('Load dataset from :',dataset_dir)
    data,label = torch.load(dataset_dir)
    print('data.shape=',data.shape,'label.shape=',label.shape)
    trainset = TensorDataset(data,label) 

    df = DataFactory(args.dataset)
    testset = df.getTestSet('target',transform_list=[transform.ToTensor(),transform.Resize(32)])

    trainloader = DataLoader(trainset, args.batchsize)
    testloader = DataLoader(testset, args.batchsize)
    criterion = torch.nn.CrossEntropyLoss()

    model = get_model(args.net, args.dataset)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    csv_list = []
    print('Training classifier')
    for e in range(args.epoch):
        train_loss, train_acc = tools.train(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = tools.test(model, testloader, criterion, device)
        csv_list.append((e,train_loss,train_acc,test_loss,test_acc))
        print(f'Epoch {e}:\nTrain loss:{train_loss:.4f} train acc:{train_acc:.4f} test loss:{test_loss:.4f} test acc:{test_acc:.4f}')

    sess = f'{args.net}_{args.dataset}_{args.eps}'
    csv_path = tools.save_csv(sess, csv_list, os.path.join(pwd,'..','..','exp',FUNC_NAME))
    net_path = tools.save_net(sess, model, os.path.join(pwd, '..', '..', 'trained_net', FUNC_NAME))

    ent = sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path, extra=args.extra)
    sqlite_proxy.rpc_insert_net(ent)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--net', type=str, default='resnet', help='Model architecture for classifier')
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--eps', type=float, default=1)
    parser.add_argument('--extra', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum',type=float, default=0.9)
    args = parser.parse_args()
    main(args)
