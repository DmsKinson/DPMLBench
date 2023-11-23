# part of the code comes from https://github.com/dayu11/Differentially-Private-Deep-Learning/blob/main/vision/DP-SGD/utils.py

import torch
import numpy as np

def mixup(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
   
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_b = y[index]
    return mixed_x,  y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)