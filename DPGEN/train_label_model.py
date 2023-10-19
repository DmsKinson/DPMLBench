import sys,os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd,'..')) 

import torch
from DataFactory import DataFactory
import argparse
import tools
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transform
import os
import torch.nn as nn
import torch.optim as optim
from helper.net import MultiClassifier

def train_label_net(dataset_name):
    print('Training label model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_net = MultiClassifier()
    label_net_dir = os.path.join(pwd,'label_net',f'{dataset_name}')
    os.makedirs(label_net_dir, exist_ok=True)
    label_net_path = os.path.join(label_net_dir,f'{label_net._get_name()}.pt')

    optimizer = optim.Adam(label_net.parameters(), lr = 0.001)
    df = DataFactory(dataset_name)
    trainset = df.getTrainSet('shadow')
    testset = df.getTestSet('shadow')
    trainloader = DataLoader(trainset, args.batchsize)
    testloader = DataLoader(testset, args.batchsize)
    criterion = nn.CrossEntropyLoss()

    for e in range(args.epoch):
        train_loss, train_acc = tools.train(label_net, trainloader, optimizer, criterion, device)
        test_loss, test_acc = tools.test(label_net, testloader, criterion, device)
        print(f'Epoch {e}:\nTrain loss:{train_loss:.5f} train acc:{train_acc:.2f} test loss:{test_loss:.5f} test acc:{test_acc:.2f}')
    
    torch.save(label_net.state_dict(), label_net_path)
    return label_net

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fmnist')
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=20)
    args = parser.parse_args()

    train_label_net(args.dataset)
