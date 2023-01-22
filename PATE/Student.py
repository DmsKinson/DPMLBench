import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

class Student():
    def __init__(self, net, device):
        self.model = net

    def train(self, trainloader: DataLoader, optimizer:Optimizer, criterion, device, epochs: int = 10):

        self.model.to(device)
        for e in range(epochs):
            self.model.train()
            trainLoss = 0
            for data, label in trainloader:
                data, label = data.to(device),label.to(device)
                optimizer.zero_grad()
                output = self.model.forward(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                trainLoss += loss.item()
            print(f"student train loss in epoch {e+1} is ",trainLoss)
        return

    # @staticmethod
    # def getStuTrainLoader(oriTrainLoader, predLabels):
    #     for i, (data, _) in enumerate(iter(oriTrainLoader)):
    #         yield data, predLabels[i*len(data): (i+1)*len(data)]

    def test(self, testloader:DataLoader, criterion, device):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in tqdm(testloader):
                data, label = data.to(device),label.to(device)
                output = self.model.forward(data)
                # sum up batch loss
                test_loss += criterion(output, label).item()
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(label.view_as(pred)).sum().item()

        test_loss /= len(testloader.dataset)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(testloader.dataset),
                100.0 * correct / len(testloader.dataset),
            )
        )
        return correct / len(testloader.dataset)
