import os,sys
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd, '..', '..'))

import torch
from torch.utils.data import TensorDataset,DataLoader
import torchvision.transforms as transform
from models import get_model
import argparse
import tools
from DataFactory import DataFactory
import sqlite_proxy

FUNC_NAME = 'private-set'

DATASET_MAP = {
    'mnist':'MNIST',
    'fmnist':'FashionMNIST',
    'svhn':'SVHN',
    'cifar10':'CIFAR10'
}

def main(args):
    device = torch.device('cuda' if(torch.cuda.is_available()) else 'cpu')
    tools.set_rng_seed(args.seed)
    dataset = DATASET_MAP[args.dataset]
    dataset_path = os.path.join(pwd,'results',dataset,f'{args.eps:.1f}',f'res_DC_{dataset}_ConvNet_10spc.pt')
    # dataset_path = '/data2/zmh/workplace/Private-Set/results/MNIST/default/res_DC_MNIST_ConvNet_1spc.pt'
    assert os.path.exists(dataset_path), f'{dataset_path} is not existed.'
    
    print('Load dataset from :',dataset_path)
    result = torch.load(dataset_path)
    images,labels = result['data'][0]
    print('data.shape=',images.shape,'label.shape=',labels.shape)
    if(images.shape[-1] != 32):
        images = transform.Resize(32)(images)
    trainset = TensorDataset(images,labels) 
    df = DataFactory(args.dataset)
    testset = df.getTestSet('target',transform_list=[transform.ToTensor(),transform.Resize(32)])

    trainloader = DataLoader(trainset, args.batchsize)
    testloader = DataLoader(testset, args.batchsize)
    criterion = torch.nn.CrossEntropyLoss()

    model = get_model(args.net, args.dataset)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.001)
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
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--eps', type=float, default=1)
    parser.add_argument('--extra', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum',type=float, default=0.9)
    args = parser.parse_args()
    main(args)