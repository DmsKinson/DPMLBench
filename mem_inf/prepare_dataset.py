import pathlib
from opacus import GradSampleModule
import torch
import argparse
import sys
pwd = sys.path[0]
sys.path.append(pwd+'/../.')
from DataFactory import DataFactory
from torch.utils.data.dataset import Subset, Dataset
from torch.utils.data.dataloader import DataLoader
import pickle
import tools
from db_models import DB_Model
from sqlite_proxy import TYPE_SHADOW
from tqdm import tqdm

class DatasetWithMember(Dataset):
    def __init__(self, memberset, nonmemberset) -> None:
        super().__init__()
        self.dataset = memberset + nonmemberset
        self.members = torch.cat([torch.ones(len(memberset)), torch.zeros(len(nonmemberset))], dim=0).long()

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1], self.members[index]

    def __len__(self, ):
        return len(self.dataset)

class BlackBoxGetter():
    def _get_data(self,shadow_model, inputs, targets, device):
        result = shadow_model(inputs)
        output, _ = torch.sort(result, descending=True)
        _, predicts = result.max(1)
        prediction = predicts.eq(targets).float()

        return output, prediction

class WhiteBoxGetter():
    def __init__(self) -> None:
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def _get_data(self,shadow_model, inputs, targets, device):
        shadow_model.train()
        shadow_model.zero_grad()
        outputs = shadow_model(inputs)
        # outputs = F.softmax(outputs, dim=1)
        losses = self.criterion(outputs, targets)
        mean_loss = losses.mean()
        mean_loss.backward()

        r_named_parameters = reversed(list(shadow_model.named_parameters())) 
        for name, parameter in r_named_parameters:
            # gradient of linear layer
            if 'weight' in name and len(parameter.shape) == 2:
                gradients = parameter.grad_sample.detach().clone().unsqueeze(1) # [column[:, None], row].resize_(100,100)
                break

        
        # make target to ont-hot label
        labels = torch.zeros(len(targets),10, device=device)
        labels.scatter_(1,targets.unsqueeze(1),1)
        losses = losses.detach().clone().unsqueeze(1)
        return outputs, losses, gradients, labels

class WhiteBoxOldGetter():
    def __init__(self) -> None:
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def _get_data(self, shadow_model, inputs, targets, device):
        shadow_model.train()
        results = shadow_model(inputs)
        # outputs = F.softmax(outputs, dim=1)
        losses = self.criterion(results, targets)

        gradients = []
        
        for loss in losses:
            loss.backward(retain_graph=True)

            gradient_list = reversed(list(shadow_model.named_parameters()))

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

class AttacksetGenerator():

    Getter_dict = {
        'black':BlackBoxGetter,
        'white':WhiteBoxGetter,
    }

    def __init__(self, net, dataset, eps, attack_type, is_uda=False) -> None:
        self.attack_set_dir = pathlib.Path(pwd).joinpath('attack_dataset',attack_type)
        self.dataset = dataset
        self.net = net
        self.eps = eps
        self.attack_type = attack_type
        self.getter = __class__.Getter_dict[attack_type]()
        self.batch_size = 8
        self.is_uda = is_uda
        if(is_uda):
            self.attack_set_dir = self.attack_set_dir.joinpath('uda')
        self.shadow_model = tools.load_model(
            DB_Model.net==net, 
            DB_Model.eps==eps, 
            DB_Model.dataset==dataset, 
            DB_Model.type==TYPE_SHADOW,
            DB_Model.extra==('uda' if is_uda else None)
        )
        if(attack_type == 'white'):
            self.shadow_model = GradSampleModule(self.shadow_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if(not self.attack_set_dir.is_dir()):
            self.attack_set_dir.mkdir(parents=True)

    def prepare_dataset(self):
        print('Prepare dataset')
        self.attack_set_dir.mkdir(parents=False, exist_ok=True)
        self.trainpath = self.attack_set_dir.joinpath(f'{self.attack_type}_{self.net}_{self.dataset}_{self.eps}.pt')
        df = DataFactory(self.dataset)
        if(self.is_uda):
            memberset = df.getUdaTrainset('shadow')
            nonmemberset = df.getUdaTestset('shadow')
        else:
            memberset = df.getTrainSet('shadow')
            nonmemberset = df.getTestSet('shadow')
        # make member and non-member has same length
        length = min(len(memberset), len(nonmemberset))
        print('length per set is',length)
        memberset = Subset(memberset, range(length))
        nonmemberset = Subset(nonmemberset, range(length))
        attackset = DatasetWithMember(memberset, nonmemberset)
        self.save_attackset(self.trainpath, attackset)

    def save_attackset(self, trainpath, dataset):
        '''
        save attack train set with shadow model
        
        Args:
            trainpath: path to save train dataset
            dataset: DatasetWithMember of target dataset
        '''
        print('Saving attackset')
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.shadow_model = self.shadow_model.to(self.device)
        self.shadow_model.eval()

        with open(trainpath, "wb") as f:
            for inputs, targets, members in tqdm(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output_bundle = self.getter._get_data(self.shadow_model, inputs, targets, self.device)
                pickle.dump((*output_bundle,members), f) 
    
def main(args):
    AttacksetGenerator(args.net, args.dataset, args.eps, args.type, args.uda).prepare_dataset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='black', type=str, choices=['black','white'])
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to run training",)
    parser.add_argument("--net", type=str, default="simplenn",)
    parser.add_argument("--seed", type=int, default=11337, help="Seed")
    # Privacy
    parser.add_argument('--eps', default=None, type=float, help='eps of target model')
    parser.add_argument('--uda', default=False, type=bool, help='indicate whether generate uda attack set')
    args = parser.parse_args()
    main(args)