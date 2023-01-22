from optparse import Values
import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import pandas as pd
from db_models import db
import itertools
import numpy as np
from db_models import DB_Csv
from name_map import * 
import matplotlib.pyplot as plt

nets = ['simplenn','simplenn_norm']
datasets = ['fmnist','cifar10']
funcs = ['relu','tanh']
eps = [0.2,0.5,1,4,8,100]

display_datasets = [DATASETS_MAP[dataset] for dataset in datasets]
display_nets = ['w/o GroupNorm','with GroupNorm']
display_eps = [str(e) for e in eps]
display_funcs = ['DP-SGD(ReLU)','DP-SGD(Tanh)']

ents = DB_Csv.select().where(
    DB_Csv.func << funcs,
    DB_Csv.net << nets,
    DB_Csv.extra == None,
)

DEFAULT_COL = ['epoch','train_loss','train_acc','test_loss','test_acc','train_cost','test_cost']
values = []
for ent in ents:
    csv = pd.read_csv(ent.location,names=DEFAULT_COL)
    train_loss = csv['train_loss'].iat[-1]
    train_acc = csv['train_acc'].iat[-1]
    test_loss = csv['test_loss'].iat[-1]
    test_acc = csv['test_acc'].iat[-1]
    value = [
        ent.func,
        ent.net,
        ent.dataset,
        ent.eps,
        train_loss,
        train_acc,
        test_loss,
        test_acc
    ]
    values.append(value)

df = pd.DataFrame(data=values,columns=['func','net','dataset','eps','train_loss','train_acc','test_loss','test_acc'])

values = []
for func in funcs:
    for net in nets:
        value = []
        for dataset in datasets:
            for e in eps:
                # print(net,dataset,e)
                # print(df[(df['func']=='relu') & (df['net']==net) & (df['dataset']==dataset) & (df['eps']==e)]['test_acc'])
                # assert False
                try:
                    test_acc = df[(df['func']==func) & (df['net']==net) & (df['dataset']==dataset) & (df['eps']==e)]['test_acc'].item()
                except:
                    print(func,net,dataset,e)
                    print(df[(df['func']==func) & (df['net']==net) & (df['dataset']==dataset) & (df['eps']==e)]['test_acc'])
                    assert False
                value.append(test_acc)
        values.append(value)

cidx = pd.MultiIndex.from_arrays([
    [*[s for dataset in display_datasets for s in itertools.repeat(dataset,len(eps))]],
    [*(display_eps*len(datasets))],
])

iidx = pd.MultiIndex.from_arrays([
    [*[s for func in display_funcs for s in itertools.repeat(func,len(funcs))]],
    [*(display_nets*len(funcs))],
])

df = pd.DataFrame(
    data=values,columns=cidx,index=iidx
)

plt.figure()
df.T.plot(kind='line',logx=True)
# print(df.T['DP-SGD(ReLU)']['w/o GroupNorm']['FMNIST'])
# plt.plot(x=eps,y=,scalex='log')
plt.savefig(os.path.join(pwd,'test.png'))
assert False
styler = df.style
styler = styler.format(na_rep='-',precision=2)

styler.to_latex(
    buf=os.path.join(pwd,'simple_norm.tex'),
    column_format='c'*(len(eps)*len(datasets)+2),
    hrules=True,
    multicol_align='c',
    clines='skip-last;data',
    convert_css=True,
)
