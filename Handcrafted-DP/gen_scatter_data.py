import csv
import os
import pickle
import sys
pwd = sys.path[0]
sys.path.append(f'{pwd}/..')

import torch 
import torchvision.transforms as transform
from torch.utils.data.dataloader import DataLoader
from kymatio.torch import Scattering2D
from DataFactory import DataFactory,TRANSFORM_DICT

import argparse
import tools
from tqdm import tqdm
from time import time

ORI_IMG_SIZE = 32*4

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tools.set_rng_seed(args.seed)
    print('Generate trainset')
    gen_dataset(args.dataset, 'train', device)
    print('Generate tesetset')
    gen_dataset(args.dataset, 'test', device)

def gen_dataset(dataset_name, type, device):
    k = 1 if(dataset_name in ('mnist','fmnist')) else 3
    k *= 81
    df = DataFactory(dataset_name)
    ori_trans_list = TRANSFORM_DICT[dataset_name]
    ori_trans_list.append(transform.Resize(size=[ORI_IMG_SIZE,ORI_IMG_SIZE]))
    if(type=='train'):
        dataset = df.getTrainSet('target',transform_list=ori_trans_list)
    elif(type=='test'):
        dataset = df.getTestSet('target',transform_list=ori_trans_list)
    else:
        raise Exception(f'Illegal type: {type}')

    folder = f'{pwd}/scatter_dataset/{dataset_name}_{type}'
    # if(os.path.exists(folder)):
    #     os.removedirs(folder)
    # os.makedirs(folder)

    dataloader = DataLoader(dataset, batch_size=512)
    scattering = Scattering2D(2, (ORI_IMG_SIZE, ORI_IMG_SIZE)).to(device)
    'id,file,label'
    csv_list = []
    idx = 0
    for data, label in tqdm(dataloader):
        start = time()
        data = data.to(device)
        scatter_data = scattering(data).reshape(-1,k,32,32).cpu()
        print('scattering cost:', time()-start)
        start = time()
        for s_data,l in zip(scatter_data,label):
            file = f'{folder}/{idx}.pt'
            # torch.save(s_data, file)
            csv_list.append((idx,file,l.item()))
            idx += 1
        print('save cost:',time()-start)
    with open(f'{folder}/label.csv','wt') as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist',choices=['mnist','fmnist','svhn','cifar10'])
    parser.add_argument("--seed", type=int, default=11337, help="Seed")

    args = parser.parse_args()
    main(args)
    # parser.add_argument('--')

