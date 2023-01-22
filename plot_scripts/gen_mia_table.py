import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import pandas as pd
from db_models import db
import itertools
import numpy as np
from name_map import * 

type = "label"
funcs = ['dpgen','gswgan','handcraft','tanh','loss','relu','adp_alloc','adpclip','gep','rgp','pate','knn']
nets = ['simplenn','resnet','inception','vgg']
datasets = ['fmnist','svhn','cifar10']
eps = [0.2,0.4,1,4,100,1000]
SQL_QUERY = f"SELECT func,net,dataset,eps,ma,auc,extra FROM Privacy WHERE type='{type}' AND shadow_dp=false; "

display_funcs = [FUNC_DB2TABLE[func] for func in funcs]
display_nets =  [NETS_MAP[net] for net in nets]
display_eps = [str(e) for e in eps]
display_datasets = [DATASETS_MAP[dataset] for dataset in datasets]
df = pd.read_sql(SQL_QUERY, db.connection())
# print(df)
# print(df[(df['func']=='relu') & (df['net']=='resnet') & (pd.isna(df["extra"])) & (df['dataset']=='mnist') & (np.isnan(df['eps']))])
# assert False

values = []
for dataset in datasets:
    for func in funcs:
        value = []
        for net in nets:
            try:
                baseline = df[(df['func']=='relu') & (pd.isna(df["extra"])) & (df['net']==net) & (df['dataset']==dataset) & (pd.isna(df['eps']))]['auc'].item()
            except ValueError as e:
                print(net,func,dataset)
            for e in eps:
                metric = np.nan
                raw = np.nan
                try:
                    if(func in ["pate","knn"]):
                        raw = df[(df['func']==func) & (df['net']==net) & (df["extra"]=="uda") & (df['dataset']==dataset) & (df['eps']==e)]['auc'].item()
                    else:
                        raw = df[(df['func']==func) & (df['net']==net) & (pd.isna(df["extra"])) & (df['dataset']==dataset) & (df['eps']==e)]['auc'].item()
                    metric = max(raw,0.5)
                    # metric = raw
                except ValueError:
                    print(net,func,dataset,e)
                value.append(metric)
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
styler = styler.format(na_rep='-',precision=3)

styler.to_latex(
    buf=os.path.join(pwd,f'mia_{type}.tex'),
    column_format='c||l|'+'|'.join(*[['c'*len(eps)]*len(nets)]) ,
    hrules=True,
    multicol_align='|c|',
    clines='skip-last;data',
    convert_css=True,
)
