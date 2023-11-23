import sys
import os
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd, '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import time

from rgp_models import get_model
from data_factory import DataFactory

from get_noise_variance import get_sigma

import tools
import sqlite_proxy

FUNC_NAME = 'rgp'

def clip_column(grad_sample, threshold=1.0):
    norms = torch.norm(grad_sample.reshape(grad_sample.shape[0], -1), dim=1)
    scale = torch.clamp(threshold/norms, max=1.0)
    grad_sample *= scale.reshape(-1, 1, 1)

def process_grad_sample(params, clipping=1, inner_t=0):
    n = params[0].grad_sample.shape[0]
    grad_norm_list = torch.zeros(n).cuda()
    for p in params: 
        flat_g = p.grad_sample.reshape(n, -1)
        current_norm_list = torch.norm(flat_g, dim=1)
        grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list)
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1

    for p in params:
        p_dim = len(p.shape)
        scaling = scaling.reshape([n] + [1]*p_dim)
        p.grad_sample *= scaling
        if(inner_t == 0):
            p.grad = torch.sum(p.grad_sample, dim=0)
        else:
            p.grad += torch.sum(p.grad_sample, dim=0)
        p.grad_sample.mul_(0.)

# Training
def train(epoch, net, ghost_optim, ghost_params, optimizer, dataloader, device, criterion, sigma):
    net.to(device)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        for m in net.modules():
            if(hasattr(m, '_update_weight')):
                m.is_training = True
        with torch.no_grad():
            net.decomposite_weight()

        # use multiple micro-batches
        stepsize = args.phy_bs
        inner_t = args.batchsize // stepsize
        if(args.batchsize % stepsize != 0):
            raise Exception(f'batchsize should be an integer multiple of physical batchsize: {args.phy_bs}.')
        
        loss = None
        outputs_list = []
        for t in range(inner_t):
            tiny_inputs, tiny_targets = inputs[t*stepsize:(t+1)*stepsize], targets[t*stepsize:(t+1)*stepsize]
            # avoid first dimension to be zero
            if(tiny_inputs.shape[0] == 0):
                continue

            tiny_inputs, tiny_targets = tiny_inputs.to(device), tiny_targets.to(device)
            tiny_outputs = net(tiny_inputs)
            tiny_loss = criterion(tiny_outputs, tiny_targets)
            tiny_loss.backward()
            # gradient clipping
            if(args.private):
                process_grad_sample(ghost_params, clipping=args.clipping, inner_t=t)
                
            if(loss == None):
                loss = tiny_loss.detach()/inner_t
            else:
                loss += tiny_loss.detach()/inner_t
            outputs_list.append(tiny_outputs.detach())

        # add noise for DP
        if(args.private):
            for p in ghost_params:
                p.grad /= args.batchsize
                p.grad += torch.normal(0, sigma*args.clipping/args.batchsize, size = p.shape).cuda()
        # reconstruct update
        with torch.no_grad():
            for module in net.modules():
                if(hasattr(module, 'get_full_grad')):
                    full_grad = module.get_full_grad(args.lr)
                    module.full_conv.weight.grad = full_grad

        net.update_weight()
        outputs = torch.cat(outputs_list)
        optimizer.step()
        optimizer.zero_grad()
        ghost_optim.zero_grad()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        total += targets.size(0)
        correct += predicted.eq(targets.data).float().cpu().sum()
        acc = 100.*float(correct)/float(total)
    
    if(epoch + 1 == args.warmup_epoch):
        # take a snapshot of current model for computing historical update
        net.update_init_weight()


    return (train_loss/batch_idx, acc)

def test(net, dataloader, device, criterion, ):
    net.eval()
    for m in net.modules():
        if(hasattr(m, '_update_weight')):
            m.is_training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)


            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*float(correct)/float(total)
    return (test_loss/batch_idx, acc)

def adjust_learning_rate(optimizer, epoch):
    if(epoch<0.3*args.epoch):
        decay = 1.
    elif(epoch<0.6*args.epoch):
        decay = 5.
    elif(epoch<0.8*args.epoch):
        decay = 25.
    else:
        decay = 125.
    

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr / decay
    return args.lr / decay

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tools.set_rng_seed(args.seed)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    batch_size = args.batchsize

    # Data
    print('==> Preparing data..')
    data_factory = DataFactory(args.dataset,args.data_root)
    trainset = data_factory.getTrainSet()
    testset = data_factory.getTestSet()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    print("=> creating model '{}'".format(args.net))
    sigma = 0
    net = get_model(args.net, dataset=args.dataset, rank=args.rank).to(device)
    if(args.private):
        datasize = len(trainset)
        q = args.batchsize / datasize
        sigma, cur_eps = get_sigma(q, args.epoch, args.eps, args.delta)
        print('noise standard deviation for eps = %.1f: '%cur_eps, sigma)

    criterion = nn.CrossEntropyLoss()

    params = []
    # list of low-rank parameters
    ghost_params = []

    for p in net.named_parameters():
        # we do not reparametrize linear layer because it is already low-rank
        # except right layer
        if('full' in p[0] or 'fc' in p[0]):
            params.append(p[1])
            p[1].requires_grad = False
        if('left' in p[0] or 'fc' in p[0]):
            ghost_params.append(p[1])

    for p in params:
        p.cached_grad = None

    # we use this optimizer to use the gradient reconstructed from the gradient carriers
    optimizer = optim.SGD(
        params,
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

    # dummy optimizer, we use this optimizer to clear the gradients of gradient carriers
    ghost_optimizer = optim.SGD(ghost_params, lr=1)

    num_p = 0
    for p in ghost_params:
        num_p += p.numel()

    print('number of parameters (low-rank): %.3f M'%(num_p/1000000), end=' ')

    num_p = 0
    for p in params:
        num_p += p.numel()

    print('number of parameters (full): %.3f M'%(num_p/1000000))

    csv_list = []

    for epoch in range(start_epoch, args.epoch):
        print(f'\nEpoch {epoch}:')
        t0 = time.time()
        train_loss, train_acc = train(epoch,net, ghost_optimizer, ghost_params, optimizer, trainloader, device, criterion, sigma)
        t1 = time.time()
        test_loss, test_acc = test(net, testloader, device, criterion)
        t2 = time.time()
        csv_list.append((epoch, train_loss, train_acc, test_loss, test_acc, t1-t0, t2-t1))
        print(f'Train loss:{train_loss:.5f} train acc:{train_acc} test loss:{test_loss} test acc:{test_acc} time cost:{t2-t0:.2f}s')


    sess = f"{args.net}_{args.dataset}_e{args.epoch}"
    if(args.private):
        sess += f"_eps{args.eps:.2f}"

    csv_path = tools.save_csv(sess, csv_list, os.path.join(pwd,'..','..','exp',FUNC_NAME))
    net_path = tools.save_net(sess, net, os.path.join(pwd, '..', '..', 'trained_net', FUNC_NAME))

    sqlite_proxy.insert_net(func=FUNC_NAME, net=args.net, dataset=args.dataset, eps=args.eps, other_param=vars(args), exp_loc=csv_path, model_loc=net_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--data_root', default=os.path.join(pwd,'..','..','dataset'), type=str, help='directory of dataset stored or loaded')
    parser.add_argument('--dataset', default='cifar10',type=str)
    parser.add_argument('--net', default='simplenn', type=str, help='model name')
    parser.add_argument('--seed', default=2, type=int,)
    parser.add_argument('--resume', '-r',default=False, action='store_true', help='resume from checkpoint')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
    parser.add_argument('--batchsize', default=256, type=int, help='batch size')
    parser.add_argument('--epoch', default=400, type=int, help='total number of epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='base learning rate (default=0.4)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum coeeficient')

    parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
    parser.add_argument('--eps', default=None, type=float, help='eps value')
    parser.add_argument('--width', default=1, type=int, help='model width')
    parser.add_argument('--delta', default=1e-5, type=float, help='delta value')
    parser.add_argument('--rank', default=16, type=int, help='rank of reparameterization')
    parser.add_argument('--clipping', default=1., type=float, help='clipping threshold')
    parser.add_argument('--warmup_epoch', default=-1, type=int, help='num. of epochs for warmup')
    parser.add_argument('--phy_bs',default=16, type=int, help='physical batchsize')


    args = parser.parse_args()
    main(args)
