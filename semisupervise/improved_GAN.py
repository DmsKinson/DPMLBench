import sys,os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd,'..','..'))

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.utils as vutils
from .generator import netG
from data_factory import DataFactory
from models import get_model
from torch.utils.data.dataset import Subset
from torchvision.utils import save_image

def LSE(data, dim=1):
    m,_ = torch.max(data, dim = dim)
    return m + torch.logsumexp(data - m.unsqueeze(1), dim=dim)

def test(net_d, testloader, criterion, device):
    net_d.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = net_d(data)
        test_loss += criterion(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cuda().sum()
    test_loss /= len(testloader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(testloader.dataset),100. * correct / len(testloader.dataset)))

def train(epoch,net_d, net_g, criterionD, criterionG, optimizerD, optimizerG, dataloader, semi_rate, device):
    mean_loss_label = 0
    mean_loss_unlabel = 0
    mean_loss_fake = 0
    mean_loss_d = 0
    mean_loss_g = 0
    total = 0
    correct = 0
    net_d.train()
    net_g.train()
    for idx, (data,real_label) in enumerate(dataloader):
        batchsize = real_label.size(0)
        n_query_per_batch = int(batchsize*semi_rate)
        data = data.to(device)
        data_label = data[:n_query_per_batch]
        real_label = real_label.to(device)[:n_query_per_batch]
        data_unlabel = data[n_query_per_batch:]

        optimizerD.zero_grad()
        # train with label
        out_label = net_d(data_label)
        loss_label = criterionD(out_label, real_label)
        mean_loss_label += loss_label.item()
        # compute train accuracy
        _, predicted = torch.max(out_label.data, 1)
        total += real_label.size(0)
        correct += predicted.eq(real_label.data).float().cpu().sum()
        # train with unlabel
        out_unlabel = net_d(data_unlabel)
        loss_unlabel = -torch.mean(LSE(out_unlabel),0) +  torch.mean(F.softplus(LSE(out_unlabel),1),0)
        mean_loss_unlabel += loss_unlabel.item()

        #train with fake
        noise = torch.normal(0, 1, [batchsize-n_query_per_batch,net_g.nz,1,1], requires_grad=True).cuda()    
        fake = net_g(noise)
        out_fake = net_d(fake.detach()) 
        loss_fake = torch.mean(F.softplus(LSE(out_fake),1),0)
        mean_loss_fake += loss_fake.item()

        loss_D = loss_label + loss_unlabel + loss_fake
        mean_loss_d += loss_D.item()

        loss_D.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z))) 
        ###########################
        optimizerG.zero_grad()
        ####### feature matching ########
        feature_real = net_d(data_label.detach(),feature=True)
        feature_fake = net_d(fake,feature=True)
        feature_real = torch.mean(feature_real,0)
        feature_fake = torch.mean(feature_fake,0)
        loss_G = criterionG(feature_fake, feature_real.detach())
        mean_loss_g += loss_G.item()

        ####### feature matching ########
        loss_G.backward()
        optimizerG.step()

    acc = 100.*float(correct)/float(total)
    idx += 1  # avoid idx=0 when batch size bigger than data amount
    if(epoch % 50==0):
        save_image(fake[:8,:,:,:],f'sample_{epoch}.png')
    print('Loss_label: %.4f Loss_unlabel: %.4f Loss_fake: %.4f Loss_D: %.4f Loss_G: %.4f Train_acc: %.4f'
              % (mean_loss_label/idx, mean_loss_unlabel/idx, mean_loss_fake/idx, mean_loss_d/idx, mean_loss_g/idx,acc))
    return mean_loss_d/idx, mean_loss_g/idx, acc

def main(args):
    nc = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_g = netG(args.nz,args.ngf,nc).to(device)
    # net_d = netD(nc,ndf).to(device)
    net_d = get_model(args.net,args.dataset).to(device)

    df = DataFactory(args.dataset)
    trainset = df.getTrainSet()
    trainset = Subset(trainset, range(1000))
    testset = df.getTestSet()

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize)
    
    optimizerD = optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    optimizerG = optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    
    criterionD = nn.CrossEntropyLoss() # binary cross-entropy
    criterionG = nn.MSELoss()
    fixed_noise = torch.normal(0, 1, [args.batchsize,args.nz,1,1]).to(device)    

    for epoch in range(args.epoch):
        print('\nEpoch:',epoch)
        train(epoch,net_d, net_g, criterionD, criterionG, optimizerD, optimizerG, dataloader, device)
        test(net_d, testloader, criterionD, device)
        if(epoch % args.fig_interval==0):
            fake_data = net_g(fixed_noise)
            vutils.save_image(vutils.make_grid(fake_data, normalize=True),'./fake/fake_samples_epoch_%03d.png' % epoch) 

        # do checkpointing
    # torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))


if(__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='resnet', type=str, help='network for experiment')
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
    parser.add_argument('--label_data', )  ##TODO

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--fig_interval', default=2, type=int)
    parser.add_argument('--batchsize', default=200, type=int)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--nz', default=100, type=int, help='Size of z latent vector')
    parser.add_argument('--ngf', default=32, type=int, help='Number of G output filters')

    args = parser.parse_args()
    main(args)