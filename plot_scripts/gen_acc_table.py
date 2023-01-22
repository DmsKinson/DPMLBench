import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import pandas as pd
from db_models import db
import itertools
import numpy as np
from name_map import * 

SQL_QUERY = "SELECT func,net,dataset,eps,test_acc,extra FROM Utility WHERE  type='target'; "

funcs = ['dpgen','gswgan','handcraft','tanh','loss','relu','adp_alloc','adpclip','gep','rgp','pate','knn']
nets = ['simplenn','resnet','inception','vgg']
datasets = ['mnist','fmnist','svhn','cifar10']
eps = [0.2,0.4,1,4,100,1000]

display_funcs = [FUNC_DB2TABLE[func] for func in funcs]
display_nets =  [NETS_MAP[net] for net in nets]
display_eps = [str(e) for e in eps]
display_datasets = [DATASETS_MAP[dataset] for dataset in datasets]
df = pd.read_sql(SQL_QUERY, db.connection())
# print(df[(df['func']=='relu') & (df['net']=='resnet') & (df['dataset']=='mnist') & (np.isnan(df['eps']))])
# assert False

values = []
for dataset in datasets:
    for func in funcs:
        value = []
        for net in nets:
            baseline = df[(df['func']=='relu') & (pd.isna(df["extra"])) & (df['net']==net) & (df['dataset']==dataset) & (np.isnan(df['eps']))]['test_acc'].item()
            for e in eps:
                metric = np.nan
                test_acc = np.nan
                try:
                    if(func in ["pate","knn"]):
                        test_acc = df[(df['func']==func) & (df["extra"]=="uda") & (df['net']==net) & (df['dataset']==dataset) & (df['eps']==e)]['test_acc'].item()
                    else:
                        test_acc = df[(df['func']==func) & (pd.isna(df["extra"])) & (df['net']==net) & (df['dataset']==dataset) & (df['eps']==e)]['test_acc'].item()
                    metric = (baseline-test_acc)*100/baseline
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
styler = styler.format(na_rep='-',precision=2)

styler.to_latex(
    buf=os.path.join(pwd,'accuracy.tex'),
    column_format='c||l|'+'|'.join(*[['c'*len(eps)]*len(nets)]) ,
    hrules=True,
    multicol_align='|c|',
    clines='skip-last;data',
    convert_css=True,
)
