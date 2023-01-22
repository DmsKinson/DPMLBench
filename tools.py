from opacus import GradSampleModule
import torch.nn as nn
import os
import csv
import pathlib
import torch
from opacus.optimizers.optimizer import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data.dataloader import DataLoader
import yaml
import numpy as np
import random   
from db_models import DB_Model
import config
import time
from torch.utils.data.dataset import Dataset

from models import get_model

pwd = os.path.split(os.path.realpath(__file__))[0]

TRAINED_NET_DIR = os.path.join(pwd, 'trained_net')
EXP_DIR = os.path.join(pwd, 'exp')

def save_csv(sess, csv_list, dst_dir=EXP_DIR):
    dst_dir = pathlib.Path(dst_dir)
    if not dst_dir.is_dir():
        dst_dir.mkdir(parents=True)
    csv_path = dst_dir.joinpath(f'{sess}.csv')
    with open(csv_path,mode='wt') as file:
        writer = csv.writer(file)
        writer.writerows(csv_list)
    return csv_path.as_posix()

def get_normal_module(net):
    if(isinstance(net, GradSampleModule)):
        try:
            net.del_grad_sample()
        except Exception as e:
            print('Error:',e)
        net.remove_hooks()
        net._clean_up_attributes()
        return net._module
    return net 

def save_pt(sess, net, dst_dir=TRAINED_NET_DIR):
    dst_dir = pathlib.Path(dst_dir)
    if not dst_dir.is_dir():
        dst_dir.mkdir(parents=True)
    net_path = dst_dir.joinpath(f'{sess}.pt')
    torch.save(net, net_path)
    return net_path.as_posix()

def save_net(sess, net, dst_dir=TRAINED_NET_DIR):
    net = get_normal_module(net)
    dst_dir = pathlib.Path(dst_dir)
    if not dst_dir.is_dir():
        dst_dir.mkdir(parents=True)
    net_path = dst_dir.joinpath(f'{sess}.pt')
    torch.save(net.state_dict(), net_path)
    return net_path.as_posix()

def save_attack_net(sess, net, dst_dir=TRAINED_NET_DIR):
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir,exist_ok=True)
    net_path = os.path.join(dst_dir,f'{sess}.pt')
    torch.save(net,net_path)
    return net_path

def get_arch(func, net, dataset, **kwargs):
    # load architecture
    if(func=='rgp'):
        from RGP.rgp_models import get_model as get_rgp_model
        arch = get_rgp_model(net, rank=kwargs.get('rank',16), dataset=dataset)
    elif(func=='tanh'):
        arch = get_model(net, dataset, act_func='tanh')
    elif(func=='handcraft'):
        k = 81 if dataset in ['mnist','fmnist'] else 81*3
        arch = get_model(net, dataset, in_channel=k)
    else:
        arch = get_model(net, dataset, )
    return arch 

def load_attack_net(*args):
    res = DB_Model.get_or_none(*args)
    assert res != None, 'Error: query returns no records!'
    print('==> Loading model from : ',res.location)
    assert res.host_ip == config.HOST_IP, f'model is not local:host={res.host_ip}'
    
    attack_net = torch.load(res.location)
    return attack_net

def load_model(*args):
    res = DB_Model.get_or_none(*args)
    assert res != None, 'Error: query returns no records!'
    print('==> Loading model from : ',res.location)
    assert res.host_ip == config.HOST_IP, f'model is not local:host={res.host_ip}'
    arch = get_arch(func=res.func, net=res.net, dataset=res.dataset,)
    
    state_dict = torch.load(res.location)
    arch.load_state_dict(state_dict)
    arch.eval()
    return arch

def load_target_model(func:str, net:str, dataset:str, eps:float, extra:str = None):
    return load_model(
        DB_Model.func==func,
        DB_Model.net==net,
        DB_Model.dataset==dataset,
        DB_Model.eps==eps,
        DB_Model.extra==extra
    )

def get_stat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset
    :return: (mean, std)
    '''
    print(len(train_data))
    channel = train_data[0][0].shape[0]
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, 
        pin_memory=True)
    mean = torch.zeros(channel)
    std = torch.zeros(channel)
    for X, _ in train_loader:
        for d in range(channel):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())
 
def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))

def persist_args(args,name:str,dir:str):
    dicts = vars(args)
    filepath = pathlib.Path(dir).joinpath(name+'.yml')
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(dicts, f, default_flow_style=False)

def set_rng_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

def get_gradient_size(model):
    gradient_size = []
    gradient_list = reversed(list(model.named_parameters()))
    for name, parameter in gradient_list:
        if 'weight' in name:
            gradient_size.append(parameter.shape)

    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2    

    return total

def get_linear_size(model):
    r_named_parameters = reversed(list(model.named_parameters())) 
    for name, parameter in r_named_parameters:
        # gradient of linear layer
        if 'weight' in name and len(parameter.shape) == 2:
            return parameter.shape

def test(net, testloader, criterion, device):
    net.eval()
    net = net.to(device)
    test_loss = 0
    correct = 0
    total = 0
    all_correct = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() 
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()
            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()

        acc = 100.*float(correct)/float(total)

    return (test_loss/batch_idx, acc)

def train(net:nn.Module, dataloader, optimizer, criterion, device):
    net.train()
    net = net.to(device)
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, label) in enumerate(dataloader):   
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += predicted.eq(label.data).float().cpu().sum()
        
    acc = 100.*float(correct)/float(total)
    return (train_loss/(batch_idx+1), acc)

class DatasetWithNewlable(Dataset):
    def __init__(self, data, new_label) -> None:
        super().__init__()
        self.data = data
        self.label = new_label

    def __getitem__(self, index) -> tuple():
        return (self.data[index][0],self.label[index])

    def __len__(self):
        return len(self.data)

class MemoryManagerProxy(BatchMemoryManager):
    def __init__(self, *, is_private, data_loader: DataLoader, max_physical_batch_size: int, optimizer: DPOptimizer):
        super().__init__(data_loader=data_loader, max_physical_batch_size=max_physical_batch_size, optimizer=optimizer)
        self.is_private = is_private

    def __enter__(self):
        if(self.is_private):
            return super().__enter__()
        else:
            return self.data_loader

    def __exit__(self, type, value, traceback):
        return super().__exit__(type, value, traceback)

class Print(nn.Module):
	def __init__(self, label=''):
		super().__init__()
		self.label = label

	def forward(self,x,):
		print(self.label)
		print(x.shape,end='\n\n')
		return x

def get_model_parameters_amount(model):
    n_param = 0
    for p in model.parameters():
        n_param += p.numel()
    return n_param