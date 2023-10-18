import sys
import os


pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import pandas as pd
from peewee import SqliteDatabase
import itertools
import numpy as np
from name_map import * 
import argparse

pwd = os.path.split(os.path.realpath(__file__))[0]
DATABASE_DIR = os.path.join(pwd,'..', 'database', 'merge.db')
main_db = SqliteDatabase(DATABASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--type',type=str,choices=['black','white'], default='white')
args = parser.parse_args()


attack_type = args.type
metric = 'auc'
funcs = ['dpgen','private-set','handcraft','tanh','loss','relu','adp_alloc','adpclip','gep','rgp','pate','knn']
nets = ['simplenn','resnet','inception','vgg']
datasets = ['fmnist','svhn','cifar10']

eps = [0.2,1,4,100,1000]
SQL_QUERY = f"SELECT func, net, dataset, eps, type, auc_mean AS mean, auc_std AS std FROM PrivacyStat;"

display_funcs = [FUNC_DB2TABLE[func] for func in funcs]
display_nets =  [NETS_MAP[net] for net in nets]
display_eps = [str(e) for e in eps]
display_datasets = [DATASETS_MAP[dataset] for dataset in datasets]
df = pd.read_sql(SQL_QUERY, main_db.connection())


values = []
for dataset in datasets:
    for func in funcs:
        value = []
        for net in nets:
            for e in eps:
                mean = np.nan
                std = np.nan
                try:
                    mean = df[(df['func']==func) & (df['net']==net) & (df['dataset']==dataset) & (df['eps']==e) & (df['type']==attack_type) ]['mean'].item()
                    std = df[(df['func']==func) & (df['net']==net) & (df['dataset']==dataset) & (df['eps']==e) & (df['type']==attack_type) ]['std'].item()
                except ValueError:
                    print(net,func,dataset,e)
                if(np.isnan(mean) or np.isnan(std)):
                    record = None
                else:
                    record = f'${mean:.2f}\pm{std:.2f}$'
                value.append(record)
        values.append(value)

cidx = pd.MultiIndex.from_arrays([
    [*[s for net in display_nets for s in itertools.repeat(net,len(eps))]],
    [*(display_eps*len(nets))],

])
iidx = pd.MultiIndex.from_arrays([
    [*[s for dataset in display_datasets for s in itertools.repeat(dataset,len(display_funcs))]],
    [*(display_funcs*len(datasets))],
])

df = pd.DataFrame(
    data=values,columns=cidx,index=iidx
)

styler = df.style
styler = styler.format(na_rep='-',precision=2)

styler.to_latex(
    buf=os.path.join(pwd,f'mia_{attack_type}.tex'),
    column_format='c||l|'+'|'.join(*[['c'*len(eps)]*len(nets)]) ,
    hrules=True,
    multicol_align='|c|',
    clines='skip-last;data',
    convert_css=True,
)
