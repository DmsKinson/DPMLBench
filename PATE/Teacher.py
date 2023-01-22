import sys
sys.path.append('..')
import torch
from models import get_model
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader


class Teacher():
    def __init__(self, id, net:str, dataset:str):
        self.model = get_model(net, dataset)
        self.id = id

    def name(self):
        return f"Teacher {id}"

    def train(self, trainloader: DataLoader, optimizer:Optimizer, criterion, device,epochs: int = 10):
        trainLoss = 0
        self.model.to(device)
        
        self.model.train()
        for e in range(epochs):
            # trainLoss = 0;
            for datas, labels in trainloader:
                datas, labels = datas.to(device), labels.to(device)
                optimizer.zero_grad()
                output = self.model(datas)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                # trainLoss += loss.item()
        # print(f"Teacher {self.id} total train loss on epoch {e} is {trainLoss}")    

    def predict(self, dataloader: DataLoader, device):
        outputs = torch.zeros(0, dtype=torch.long).to(device)
        self.model.eval()
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                output = self.model(images)     # softmax
                ps = torch.argmax(output, dim=1)    # label 
                outputs = torch.cat((outputs, ps.to(device))) # 1xN N:len(dataset)
        return outputs

    def test(self, dataloader:DataLoader, device):
        acc = 0.
        self.model.eval()

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                output = self.model(images)
                ps = torch.argmax(output,dim=1)
                acc += (ps == labels).sum().item()
        return acc / len(dataloader.dataset)

