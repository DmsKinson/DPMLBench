import sys,os
pwd = sys.path[0]
sys.path.append(os.path.join(pwd,'..','..'))

from genericpath import exists
import os
import torch
from torch.utils.data.dataset import Dataset
import shutil
import torch.nn as nn
import time
import math
import aggregation
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
import pathlib 
from tqdm import tqdm
from tools import get_arch
import numpy as np

def predict(net, dataloader: DataLoader, device):
    outputs =[]
    net.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            output = net(images)     # softmax
            ps = torch.argmax(output, dim=1)    # label 1xbs
            outputs.append(ps)
    outputs = torch.cat(outputs,0) # 1xN, N:len(dataset)
    return outputs

def teachers_predict(net:str,dataset:str,net_dir, stu_trainloader: DataLoader, device,) -> torch.Tensor:
    preds = []
    net_dir = pathlib.Path(net_dir)
    files = os.listdir(net_dir)
    n_teacher = 0
    files = filter(lambda x:x.endswith('teacher.pt'), files)
    
    for f in tqdm(files) :
        net_state = torch.load(net_dir.joinpath(f))
        model = get_arch(func='pate', net=net, dataset=dataset,)
        model.load_state_dict(net_state)
        model = model.to(device)
        results = predict(model, stu_trainloader, device) #1xN
        preds.append(results)
        n_teacher += 1
    preds = torch.stack(preds)  # nTeacher*N
    return preds, n_teacher

def cal_ensemble_acc(dataset:Dataset, labels, device):
    batch = 512
    loader = DataLoader(dataset,batch_size=batch)
    cnt = 0
    labels = labels.to(device)
    for idx, (_,label) in enumerate(loader): 
        label = label.to(device)
        cnt += (label == labels[idx*batch:(idx+1)*batch]).sum().item()
    return cnt/len(labels)

def test(net, testloader, criterion, device):
    net.eval()
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

def train(net:nn.Module, dataloader, optimizer, criterion, device, epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()

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
    t1 = time.time()
    return (train_loss/(batch_idx+1), acc)

def obtain_new_data(net:str, dataset:str, opt_n_teacher:int, eps:float, delta:float, whole_set:Dataset, n_query:int, device, agg_mode='LNMax'):
    n_whole_samples = len(whole_set)
    print('num of whole samples for student is',n_whole_samples, ',n_query is',n_query)

    teacher_root = os.path.join(pwd,'..','trained_net','pate',net,dataset,f'{opt_n_teacher}_teachers')
    preds_path = os.path.join(teacher_root,f'teacher_preds_{n_whole_samples}.pt')
    if(not os.path.exists(preds_path)):
        print('No teacher prediction existed.')
        stu_trainloader = DataLoader(whole_set, batch_size=100)
        assert os.path.isdir(teacher_root), f"{teacher_root} is not a dir."
        preds, n_teacher = teachers_predict(net,dataset,teacher_root, stu_trainloader, device)
        torch.save((preds, n_teacher), preds_path)
    else:
        preds, n_teacher = torch.load(preds_path)
        print(f'Load preds created by {n_teacher} teachers from: {preds_path}')
    
    assert opt_n_teacher == n_teacher, f'n_teacher is not exist. Load {n_teacher} actually'
    
    print(f'Start {agg_mode} aggregation')
    if agg_mode == "GNMax":
        raise "Invalid aggregation mode"
        # new_data, new_labels = aggregation.GNMax(stu_trainset, preds, sigma1=sigma1, sigma2=sigma2, T=0.6*n_teacher)
        
    elif agg_mode == "LNMax" :
        # sigma calculating formulation is according to the Theorem 2 in "Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data"
        # solve sigma from the following equation, where eps is taken as the known quantity 
        cal_gamma = lambda T, delta, eps:(math.sqrt(2*T*math.log(1/delta) + 4*eps*T)-math.sqrt(2*T*math.log(1/delta)) )/ (4*T)
        gamma = 0
        if(eps != None):
            # in semisupervisor setting, only $n_query using noisy aggregation mechanism, so we only accoutant $n_query quiries' privacy budget
            gamma = cal_gamma(n_query, delta, eps)
        print(f'for eps={eps}, gamma={gamma}')
        new_data, new_labels = aggregation.LNMax(whole_set, preds, gamma=gamma)
    else:
        raise "Invalid aggregation mode"
    return new_data, new_labels

def uda_dataset_process(dataset, num_labeled, num_classes = 10):
    label_per_class = num_labeled // num_classes
    labels = []
    for _,l in dataset:
        labels.append(l)
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == num_labeled

    label_set = Subset(dataset,labeled_idx)
    unlabel_set = Subset(dataset,unlabeled_idx)
    return label_set, unlabel_set

class DatasetWithNewlable(Dataset):
    def __init__(self, data, new_label) -> None:
        super().__init__()
        self.data = data
        self.label = new_label

    def __getitem__(self, index) -> tuple():
        return (self.data[index][0],self.label[index])

    def __len__(self):
        return len(self.data)

    def save(self, root):
        if exists(root) : shutil.rmtree(root)
        os.mkdir(root)
        torch.save(self,root+'/dataset.pt')