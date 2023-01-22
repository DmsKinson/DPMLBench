import numpy as np
import torch
from torch.functional import Tensor
from torch.utils.data.dataset import Dataset, Subset
from tqdm import tqdm

def LNMax(dataset:Dataset, ori_preds:Tensor, **kwargs):
    gamma = kwargs.get('gamma')
    if(gamma != 0):
        beta = 1 / kwargs.get("gamma")
        lap = torch.distributions.laplace.Laplace(0,torch.tensor(beta))
    counts_list = []
    for image_preds in tqdm(ori_preds.T):
        label_counts = torch.bincount(image_preds,minlength=10)      
        if(gamma != 0):
            # add noise
            label_counts = label_counts.to(dtype=torch.float32)
            label_counts += lap.sample(label_counts.shape).to(label_counts.device)
        counts_list.append(label_counts)
        
    noisy_counts = torch.stack(counts_list)
    # pick noisy argmax as new label
    new_labels = torch.argmax(noisy_counts,dim=1)
    return dataset, new_labels


def GNMax(dataset:Dataset, ori_preds, **kwargs):
    """
    Confident-GNMax Aggregator: given a query, consensus among teachers is first estimated 
    in a privacy-preserving way to then only reveal confident teacher predictions.

    Args:
        sigma1 (float or Tensor) : noisy parameter for noise screening
        sigma2 (float or Tensor) : noisy parameter for noise argmax
        T (int) : threshold of noise screening
    """
    sigma1 = kwargs.get("sigma1")
    sigma2 = kwargs.get("sigma2")
    T = kwargs.get("T")
    gaus1 = torch.distributions.Normal(0, sigma1)
    gaus2 = torch.distributions.Normal(0, sigma2)
    counts_list = []
    remain_data_idx= []
    for idx, image_preds in enumerate(torch.transpose(ori_preds, 0, 1)) :  #after transpose:  n_stu_dataset * n_teacher
        label_counts = torch.bincount(image_preds,minlength=10).type(torch.float64)    
        # noise screening   
        if torch.max(label_counts) + gaus1.sample() > T:
            # add noise 
            print(label_counts)
            label_counts += gaus2.sample(label_counts.shape).to(label_counts.device)
            remain_data_idx.append(idx)
            counts_list.append(label_counts)
    noisy_counts = torch.stack(counts_list)
    new_labels = torch.argmax(noisy_counts,dim=1)
    new_data = Subset(dataset,remain_data_idx)

    return new_data, new_labels

