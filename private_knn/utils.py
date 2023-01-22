from functools import partial
import torch
from skimage import color
from skimage.feature import hog
import os
import sys
import pathlib
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from autodp import anaRDPacct,rdp_bank
from torch.utils.data import DataLoader

pwd=sys.path[0]

DEFAULT_SIGMA_MIN_BOUND = 0.01
DEFAULT_SIGMA_MAX_BOUND = 10
MAX_SIGMA = 2000

SIGMA_PRECISION = 0.01

class FeatureExtractor():
    def __init__(self, root_dir, sess, device) -> None:
        self.device = device
        self.root_dir = root_dir
        self.hog_name = f'{sess}_hog.pt'
        self.model_name = f'{sess}_model.pt'

    def extract_feature_with_model(self, model, dataloader, prefix, save=True):
        real_path = os.path.join(self.root_dir, f'{prefix}_{self.model_name}')
        model.eval()
        print('Extract_feature_with_model')
        if(os.path.exists(real_path)):
            print('Load feature from',real_path)
            feature, label = torch.load(real_path)
            return feature, label
        output_list = []
        label_list = []
        with torch.no_grad():
            for data,label in tqdm(dataloader):
                data, label = data.to(self.device), label.to(self.device)
                output = model(data)
                label_list.append(label)
                output_list.append(output)
        feature = torch.cat(output_list)
        labels = torch.cat(label_list)
        if(save):
            print('Save feature', real_path)
            torch.save((feature, labels), real_path)
        return feature, labels

    def extract_feature_with_hog(self, dataset, prefix, save=True):
        real_path = f'{self.root_dir}/{prefix}_{self.hog_name}'
        print('Extract_feature_with_hog')
        if(os.path.exists(real_path)):
            print('Load feature from', real_path)
            feature, label = torch.load(real_path)
            return feature, label
        output_list = []
        label_list = []
        for data,label in tqdm(dataset):
            data = torch.permute(data, [1,2,0]).squeeze_()
            if(len(data.shape) == 3):
                data = color.rgb2gray(data)
            hog_data = hog(data, orientations=8, block_norm='L2')
            feature = torch.tensor(hog_data)
            output_list.append(feature)
            label_list.append(label)
        feature = torch.stack(output_list)
        label = torch.tensor(label_list)
        if(save):
            print('Save feature',real_path)
            torch.save((feature, label),real_path)
        return feature, label

def cal_ensemble_acc(ori_dataset:Dataset, labels, device):
    assert len(ori_dataset) == len(labels), f"ori_dataset should have the same length with labels: ori_dataset:{len(ori_dataset)},labels:{len(labels)}"
    batch = 512
    loader = DataLoader(ori_dataset,batch_size=batch)
    cnt = 0
    labels = labels.to(device)
    for idx, (_,label) in enumerate(loader): 
        label = label.to(device)
        cnt += (label == labels[idx*batch:(idx+1)*batch]).sum().item()
    return cnt/len(labels)

def get_epsilon(sigma, delta, sample_prob, n_query):
    acct = anaRDPacct()
    gaussian_mech = lambda x: rdp_bank.RDP_inde_pate_gaussian({'sigma': sigma}, x)
    acct.compose_poisson_subsampled_mechanisms(gaussian_mech, sample_prob, coeff = n_query)
    return acct.get_eps(delta)

def search_sigma(target_eps, delta, sample_prob, n_query):
    print('\nSearch sigma')
    get_eps = partial(get_epsilon,delta=delta, sample_prob=sample_prob, n_query=n_query)
    
    eps = float("inf")
    sigma_min = DEFAULT_SIGMA_MIN_BOUND
    sigma_max = DEFAULT_SIGMA_MAX_BOUND

    while eps > target_eps:
        sigma_max = 2*sigma_max
        eps = get_eps(sigma_max)
        if sigma_max > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while sigma_max - sigma_min > SIGMA_PRECISION:

        sigma = (sigma_min + sigma_max)/2
        
        eps = get_eps(sigma)
        print('sigma:',sigma,'eps:',eps)
        if eps < target_eps:
            sigma_max = sigma
        else:
            sigma_min = sigma

    return sigma, eps

