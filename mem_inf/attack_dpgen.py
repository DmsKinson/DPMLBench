import logging
import pathlib
import sys
import os
from torchvision import transforms
from opacus import GradSampleModule

from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+'/../')

# for loading GEP model
sys.path.append(pwd+'/../GEP')
import basis_matching

from db_models import DB_Model

import torch 
import sqlite_proxy
import argparse
from DataFactory import DataFactory
from sklearn import metrics
import tools
import config
from AttackFactory import get_attack
from data_manager import get_md5
import numpy as np

torch.set_warn_always(False)

logger = logging.getLogger('attack')

def save_posterior(infer, attack_type:str, func:str, net:str, dataset:str, eps:float, auc:float,shadow_dp:bool, extra:str):
    mia_dir = pathlib.Path(pwd).joinpath('..','exp','attack',attack_type)
    if(shadow_dp):
        mia_dir = mia_dir.joinpath('shadow_dp')
    mia_dir = mia_dir.joinpath(func)
    mia_dir.mkdir(parents=True, exist_ok=True)
    sess = f'{attack_type}_{net}_{dataset}_{eps}_{shadow_dp}'
    if(extra != None):
        sess += extra
    posterior_path = tools.save_pt(sess, infer, dst_dir=mia_dir.as_posix())
    posterior_checksum = get_md5(posterior_path)

    ent = sqlite_proxy.insert_mia(
        type=attack_type,
        func=func,
        net=net,
        dataset=dataset,
        eps=eps,
        prob_loc=posterior_path,
        prob_checksum=posterior_checksum,
        auc=auc,
        host_ip=config.HOST_IP,
        shadow_dp=shadow_dp,
        extra=extra,
    )
    # sqlite_proxy.rpc_insert_mia(ent)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tools.set_rng_seed(args.seed)

    # attack model and shadow model use 'eps column' to distinguish whether using dp shadow models, none for no-dp
    eps_for_load = args.eps if args.shadow_dp else None
    attack = get_attack(func=args.func,type=args.type, net=args.net, eps=eps_for_load, extra=args.extra, dataset=args.dataset, device=device,)
    print('Load shadow model.')
    shadow_model = tools.load_model(
        DB_Model.type==sqlite_proxy.TYPE_SHADOW,
        DB_Model.net==args.net,
        DB_Model.dataset==args.dataset,
        DB_Model.eps==eps_for_load,
        DB_Model.extra==args.extra,
    )
    if(attack.is_attack_model_exist()):
        print('Load attack model.')
        attack.load_attack_model()
    else:
        print('No existing attack model.')
        attack.prepare(shadow_model=shadow_model)
        print('Loading attack dataset...')
        attack.check_dataset()
        print('==> Training attack model')
        for epoch in range(args.epoch):
            print('Epoch ',epoch)
            loss, acc = attack.train()
        # save attack model for every (net, dataset)
        attack.save_attack_model()
    
    # dataset for training target model
    df = DataFactory(args.dataset)
    
    target_member = df.getTrainSet(mode='target',)
    non_target_member = df.getTestSet(mode='target',)
    # clip data length member data and non-member data to the same 
    if(args.type == 'label'):
        infer_length = 100
    else:
        infer_length = min(len(target_member), len(non_target_member))
    target_member, non_target_member = Subset(target_member, list(range(infer_length))), Subset(non_target_member, list(range(infer_length)))
    target_mem_loader = DataLoader(target_member, batch_size=8)
    target_nonmem_loader = DataLoader(non_target_member, batch_size=8)

    # infer target model
    print('Load target model.')
    # target_model = tools.load_model(
    #     DB_Model.func==args.func, 
    #     DB_Model.net==args.net, 
    #     DB_Model.dataset==args.dataset, 
    #     DB_Model.eps==args.eps, 
    #     DB_Model.extra==args.extra,
    #     DB_Model.type==sqlite_proxy.TYPE_TARGET)

    target_model = tools.get_arch(args.func, args.net, args.dataset)
    path = f'{pwd}/../trained_net/DPGEN/{args.net}_{args.dataset}_{args.eps}.pt'
    target_model.load_state_dict(torch.load(path))

    if(args.type=='white'):
        target_model = GradSampleModule(target_model)
    elif(args.type=='label'):
        print('Calibrating threshold')
        x_train, y_train = next(iter(DataLoader(df.getTrainSet('shadow'), 100, shuffle=True)))
        x_test, y_test = next(iter(DataLoader(df.getTestSet('shadow'), 100, shuffle=True)))
        x_train, y_train = x_train.numpy(), y_train.numpy()
        x_test, y_test = x_test.numpy(), y_test.numpy()
        attack.calibrate_threshold(shadow_model, x_train, y_train, x_test, y_test)
    print('Start inferring')
    member_probs = attack.infer(target_model, target_mem_loader)
    if(isinstance(member_probs,tuple)):
        member_probs = member_probs[0]
        
    nonmember_probs = attack.infer(target_model, target_nonmem_loader)
    if(isinstance(nonmember_probs,tuple)):
        nonmember_probs = nonmember_probs[0]

    infered_probs = torch.cat([member_probs, nonmember_probs]).cpu()
    y_true = torch.cat([torch.ones(len(target_member)), torch.zeros(len(non_target_member))], dim=0).cpu()
    
    if(np.isnan(infered_probs).all()):
        print('nan in "infered_probs"')
        auc = 0.5
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_score=infered_probs, y_true=y_true)
        auc = metrics.auc(fpr,tpr)
    print(f'auc={auc}')

    # save infer record
    # save_posterior(infer=(infered_probs, y_true), attack_type=args.type, func=args.func, net=args.net, dataset=args.dataset, eps=args.eps, auc=auc, shadow_dp=args.shadow_dp, extra=args.extra)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--func', default='relu', type=str, choices=[
        'relu','tanh','loss','adp_alloc','gep','rgp','handcraft','lp-2st','alibi','pate','adpclip','knn','dpgen'
    ])
    parser.add_argument('--type', default='white', type=str, choices=['black','white','label','white_old'])
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to run training",)
    parser.add_argument("--net", type=str, default="simple",)
    parser.add_argument("--seed", type=int, default=11337, help="Seed")
    parser.add_argument("--epoch", type=int, default=50, help="epochs to train attack model")
    # Privacy
    parser.add_argument('--eps', default=None, type=float,help='eps of target model')
    parser.add_argument('--shadow_dp', action='store_true',help='whether shadow model needs to be trained with dp')
    parser.add_argument('--extra', default=None, type=str, help='extra field, default value is None')
    args = parser.parse_args()
    # print(args.eps)
    main(args)