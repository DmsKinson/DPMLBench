import logging
import sys
import os

import numpy as np
from opacus import GradSampleModule
from kymatio.torch import Scattering2D

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd,'..'))

import pickle
from DataFactory import DataFactory
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset,Subset
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.inference.membership_inference.label_only_boundary_distance import LabelOnlyDecisionBoundary
from label_only_boundary_attack import LabelOnlyAttack
import torch.optim as optim
import pathlib
from models import get_attack_model
import torch.nn.functional as F
from typing import Tuple
import tools
from db_models import DB_Model
import sqlite_proxy
from data_manager import get_md5
import config

attack_logger = logging.getLogger('MIAttack')

TYPE_BLACK = 'black'
TYPE_WHITE = 'white'
TYPE_LABEL = 'label'
TYPE_WHITE_OLD = 'white_old'

class DatasetWithMember(Dataset):
    def __init__(self, memberset, nonmemberset) -> None:
        super().__init__()
        self.dataset = memberset + nonmemberset
        self.members = torch.cat([torch.ones(len(memberset)), torch.zeros(len(nonmemberset))], dim=0).long()

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1], self.members[index]

    def __len__(self, ):
        return len(self.dataset)

class MembershipInference():

    def __init__(self, func, net, dataset, eps, extra, attack_type:str, device) -> None:
        self.epochs = 50
        self.batch_size = 64
        self.learning_rate = 1e-5
        self.attack_model = None
        self.func = func
        self.device = device
        self.dataset = dataset
        self.net = net
        self.eps = eps
        self.extra = extra
        self.attack_set_dir = pathlib.Path(pwd).joinpath('attack_dataset',attack_type)
        self.attack_model_dir = pathlib.Path(pwd).joinpath('..','trained_net','attack',attack_type)
        if(self.extra == 'uda'):
            self.attack_set_dir = self.attack_set_dir.joinpath('uda')
            self.attack_model_dir = self.attack_model_dir.joinpath('uda')
        self.attack_set_path = ''
        self.criterion = nn.CrossEntropyLoss()
        self.attack_type = attack_type

    def prepare(self,**kwargs):
        pass

    def infer(self, target_model, dataloader):
        raise NotImplementedError

    def train(self) -> Tuple[float, float]:
        raise NotImplementedError

    def is_attack_model_exist(self):
        self.attack_query = DB_Model.get_or_none(
            DB_Model.func == self.attack_type,
            DB_Model.net == self.net, 
            DB_Model.dataset==self.dataset,
            DB_Model.eps == self.eps,
            DB_Model.type==sqlite_proxy.TYPE_ATTACK,
            DB_Model.extra==self.extra,
            )
        if(self.attack_query == None):
            return False
        if(self.attack_query.host_ip != config.HOST_IP):
            attack_logger.error(f'model is not local:host={self.attack_query.host_ip}')
            return False 
        return True

    def load_attack_model(self):
        print('==> Loading model from : ',self.attack_query.location)
        self.attack_model = torch.load(self.attack_query.location)
        self.attack_model.eval()

    def save_attack_model(self):
        self.attack_model_dir.mkdir(parents=True, exist_ok=True)
        sess = f'{self.attack_type}_{self.net}_{self.dataset}_{self.eps}'
        model_path = tools.save_attack_net(sess, self.attack_model, dst_dir=self.attack_model_dir.as_posix())
        model_checksum = get_md5(model_path)

        ent = sqlite_proxy.insert_net(
            func=self.attack_type,
            net=self.net,
            eps=self.eps,
            dataset=self.dataset,
            model_loc=model_path,
            model_checksum=model_checksum,
            model_type=sqlite_proxy.TYPE_ATTACK,
            host_ip=config.HOST_IP,
            extra=self.extra,
            )
        sqlite_proxy.rpc_insert_net(ent)

    def _get_data(self):
        raise NotImplementedError

    def check_dataset(self):
        filename = f'{self.attack_type}_{self.net}_{self.dataset}_{self.eps}.pt'
        self.attack_set_path = self.attack_set_dir.joinpath(filename)
        if(not self.attack_set_path.is_file()):
            raise Exception(f'attack dataset is not existed:{self.attack_set_path.as_posix()}')
        print(f'Load attack dataset from: {self.attack_set_path.as_posix()}')

class MembershipInferenceBlackBox(MembershipInference):
    def __init__(self, func, net, dataset, extra, eps, device, **kwargs) -> None:
        super().__init__(func,net, dataset, eps, extra, TYPE_BLACK, device, **kwargs)
        if(func == 'handcraft'):
            self.scattering = Scattering2D(2, [128,128]).to(self.device)
            self.K = 81 if(dataset in ['mnist','fmnist']) else 81*3

    def prepare(self, **kwargs):
        self.attack_model = get_attack_model('black',)
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=self.learning_rate) 

    def _get_data(self, model, inputs, targets):
        if(self.func == 'handcraft'):
            with torch.no_grad():
                inputs = self.scattering(inputs).view(-1,self.K,32,32)
        result = model(inputs)
        output, _ = torch.sort(result, descending=True)
        _, predicts = result.max(1)
        prediction = predicts.eq(targets).float()

        return output, prediction
    
    def train(self):
        self.attack_model = self.attack_model.to(self.device) 
        self.attack_model.train()  
     
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0
        with open(self.attack_set_path, "rb") as f:
            while(True):
                try:
                    output, prediction, members = pickle.load(f)
                    output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)
                    prediction = torch.unsqueeze(prediction,1)
                    
                    self.optimizer.zero_grad()
                    results = self.attack_model(output, prediction)
                    results = F.softmax(results, dim=1)

                    losses = self.criterion(results, members)
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    batch_idx += 1
                except EOFError:
                    break
        print( f'Train Acc={100.*correct/total:.3f}% ({correct}/{total}), Loss={train_loss/batch_idx:.3f}')
        return train_loss/batch_idx, 100.*correct/total

    def infer(self, target_model, dataloader):
        self.attack_model.eval()
        target_model.eval()
        target_model = target_model.to(self.device)

        posterior_list = []
        label_list = []
        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(self.device), label.to(self.device)
                output, prediction = self._get_data(target_model, data, label)
                prediction = torch.unsqueeze(prediction,1)

                results = self.attack_model(output, prediction)
                results = F.softmax(results, dim=1)
                # only needs posterior of being member
                label_list.append(label.cpu())
                posterior_list.append(results[:,1])
            posterior = torch.cat(posterior_list, dim=0)
            labels = torch.cat(label_list, dim=0)
        return posterior, labels

class MembershipInferenceWhiteBox(MembershipInference):
    def __init__(self, func, net, dataset, eps, extra, device) -> None:
        super().__init__(func, net, dataset, eps, extra, TYPE_WHITE, device)
        # loss function for obtain target/shadow model's gradient
        self.target_criterion = nn.CrossEntropyLoss(reduction='none')
        # loss function for training attack model
        self.criterion = nn.BCEWithLogitsLoss()
        if(func == 'handcraft'):
            self.scattering = Scattering2D(2, [128,128]).to(self.device)
            self.K = 81 if(dataset in ['mnist','fmnist']) else 81*3

    def prepare(self, **kwargs):
        shadow_model = kwargs['shadow_model']
        linear_shape = tools.get_linear_size(shadow_model)    
        self.attack_model = get_attack_model('white_test', kernel_size= linear_shape[0], layer_size=linear_shape[1])
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=self.learning_rate) 
    
    def _get_data(self, model, inputs, targets):
        model.train()
        model.zero_grad()
        if(self.func == 'handcraft'):
            with torch.no_grad():
                inputs = self.scattering(inputs).view(-1,self.K,32,32)
        outputs = model(inputs)
        # outputs = F.softmax(outputs, dim=1)
        losses = self.target_criterion(outputs, targets)
        mean_loss = losses.mean()
        mean_loss.backward()

        r_named_parameters = reversed(list(model.named_parameters())) 
        for name, parameter in r_named_parameters:
            # gradient of linear layer
            if 'weight' in name and len(parameter.shape) == 2:
                gradients = parameter.grad_sample.detach().clone().unsqueeze(1) # [column[:, None], row].resize_(100,100)
                break

        
        # make target to ont-hot label
        labels = torch.zeros(len(targets),10,device=self.device)
        labels.scatter_(1,targets.unsqueeze(1),1)
        losses = losses.detach().clone().unsqueeze(1)
        return outputs, losses, gradients, labels

    def train(self):
        self.attack_model = self.attack_model.to(self.device)
        self.attack_model.train()  
     
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0
        with open(self.attack_set_path, "rb") as f:
            while(True):
                try:
                    output, loss, gradients, label, members = pickle.load(f)
                    output, loss, gradients, label, members = output.to(self.device), loss.to(self.device), gradients.to(self.device), label.to(self.device), members.to(self.device)
                    
                    outputs = self.attack_model(output=output, loss=loss, gradient=gradients, label=label)
                    losses = self.criterion(outputs, members.unsqueeze(1).float())
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    predicted = (torch.sigmoid(outputs.squeeze(1))>0.5).long()
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()
                    batch_idx += 1
                except EOFError:
                    break
        print( f'Train Acc={100.*correct/total:.3f}% ({correct}/{total}), Loss={train_loss/batch_idx:.3f}')
        return train_loss/batch_idx, 100.*correct/total

    def infer(self, target_model, dataloader):

        self.attack_model.eval()
        # require gradient but don't update
        target_model = target_model.to(self.device) 
        target_model.train()

        posterior_list = []
        label_list = []
        for data, label in dataloader:
            data, label = data.to(self.device), label.to(self.device)
            output, loss, gradient, label = self._get_data(target_model, inputs=data, targets=label)
            output, loss, gradient, label = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device)

            results = self.attack_model(output=output, loss=loss, gradient=gradient, label=label)
            results = torch.sigmoid(results)

            # only needs posterior of being member
            posterior_list.append(results.detach())
            label_list.append(label.cpu())
        posterior = torch.cat(posterior_list, dim=0)
        labels = torch.cat(label_list, dim=0)
        return posterior, labels

class MembershipInferenceLabelOnly(MembershipInference):
    def __init__(self, func, net, dataset, eps, extra, device) -> None:
        super().__init__(func, net, dataset, eps, extra, TYPE_LABEL, device)
        self.channel = 1 if self.dataset in ['mnist','fmnist'] else 3
        self.distance_threshold = None      
        if(func == 'handcraft'):
            self.scattering = Scattering2D(2, [32*4,32*4]).to(self.device)
            self.K = 81 if(dataset in ['mnist','fmnist']) else 81*3    

    def calibrate_threshold(self, shadow_model, x_train, y_train, x_test, y_test):
        art_shadow = PyTorchClassifier(model=shadow_model, loss=self.criterion, input_shape=(self.channel,32,32,), channels_first=True, nb_classes=10,clip_values=(-3,3))
        mia_label_only = LabelOnlyAttack(art_shadow)
        mia_label_only.calibrate_distance_threshold(x_train, y_train, x_test, y_test)
        self.distance_threshold = mia_label_only.distance_threshold_tau
        print('distance threshold=',self.distance_threshold)

    def infer(self, target_model, dataloader):
        channel = self.channel
        if(self.func == 'handcraft'):
            channel = self.channel*81
        art_model = PyTorchClassifier(model=target_model, loss=self.criterion, input_shape=(channel,32,32,), channels_first=True, nb_classes=10,clip_values=(-3,3))
        mia_label_only = LabelOnlyAttack(art_model, distance_threshold_tau=self.distance_threshold)
        # save threshold to reuse in next inference
        posteriors = []
        label_list = []
        for data, label in dataloader:
            if(self.func == 'handcraft'):
                data = data.to(self.device)
                data = self.scattering(data).reshape(-1,channel,32,32).cpu()
            label_list.append(label)
            data, label = data.numpy(), label.numpy()
            pred_posterior = mia_label_only.infer(data, label, probabilities=True)
            pred_posterior = torch.from_numpy(pred_posterior)
            posteriors.append(pred_posterior[:,1])
        posteriors = torch.cat(posteriors, dim=0)
        labels = torch.cat(label_list, dim=0)
        return posteriors, labels

    def is_attack_model_exist(self):
        # no need for prepare
        return True

    def load_attack_model(self):
        attack_logger.info("label-only attack doesn't need loading model")
    
    
# deprecated
class MembershipInferenceWhiteOld(MembershipInference):
    def __init__(self, func, net, dataset, eps, extra, device) -> None:
        super().__init__(func, net, dataset, eps, extra, TYPE_WHITE_OLD, device)
        self.target_criterion = nn.CrossEntropyLoss(reduction='none')

    def prepare(self, **kwargs):
        shadow_model = kwargs['shadow_model']
        total = tools.get_gradient_size(shadow_model)
        self.attack_model = get_attack_model('white', total= total)
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=self.learning_rate) 
    
    def _get_data(self, model:nn.Module, inputs, targets):
        model.train()
        results = model(inputs)
        # outputs = F.softmax(outputs, dim=1)
        losses = self.target_criterion(results, targets)

        gradients = []
        
        for loss in losses:
            loss.backward(retain_graph=True)

            gradient_list = reversed(list(model.named_parameters()))

            for name, parameter in gradient_list:
                if 'weight' in name:
                    gradient = parameter.grad.clone() # [column[:, None], row].resize_(100,100)
                    gradient = gradient.unsqueeze_(0)
                    gradients.append(gradient.unsqueeze_(0))
                    break

        labels = []
        for num in targets:
            label = [0 for i in range(10)]
            label[num.item()] = 1
            labels.append(label)

        gradients = torch.cat(gradients, dim=0)
        losses = losses.unsqueeze_(1).detach()
        outputs, _ = torch.sort(results, descending=True)
        labels = torch.Tensor(labels)

        return outputs, losses, gradients, labels

    def train(self):
        self.attack_model = self.attack_model.to(self.device)
        self.attack_model.train()  
     
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0
        with open(self.attack_set_path, "rb") as f:
            while(True):
                try:
                    output, loss, gradient, label, members = pickle.load(f)
                    output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)

                    results = self.attack_model(output=output, loss=loss, gradient=gradient, label=label)
                    # results = F.softmax(results, dim=1)
                    losses = self.criterion(results, members)
                    
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    batch_idx += 1
                except EOFError:
                    break
        print( f'Train Acc={100.*correct/total:.3f}% ({correct}/{total}), Loss={train_loss/batch_idx:.3f}')
        return train_loss/batch_idx, 100.*correct/total

    def infer(self, target_model, dataloader):
        self.attack_model.eval()
        # require gradient but don't update
        target_model = target_model.to(self.device) 

        posterior_list = []
        for data, label in dataloader:
            data, label = data.to(self.device), label.to(self.device)
            output, loss, gradient, label = self._get_data(target_model, inputs=data, targets=label)
            output, loss, gradient, label = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device)

            results = self.attack_model(output=output, loss=loss, gradient=gradient, label=label)
            results = F.softmax(results, dim=1)

            # only needs posterior of being member
            posterior_list.append(results[:,1].detach())
        posterior = torch.cat(posterior_list, dim=0)
        return posterior