import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import pandas as pd
from db_models import db
import itertools
import numpy as np
from name_map import * 

SQL_QUERY = "SELECT id, func, net, dataset, eps, type, train_acc, test_acc FROM Utility WHERE (func='rgp' AND eps is NULL) or (func='relu' AND eps is NULL) ORDER BY net,dataset"

funcs = ['rgp','relu']
nets = ['simplenn','resnet','inception','vgg']
datasets = ['mnist','fmnist','svhn','cifar10']

display_funcs = ['RGP(w/o DP)', 'SGD']
display_nets =  [NETS_MAP[net] for net in nets]
display_datasets = [DATASETS_MAP[dataset] for dataset in datasets]
df = pd.read_sql(SQL_QUERY, db.connection())
values = []
for dataset in datasets:
    value = []
    for net in nets:
        for func in funcs:
            test_acc = df[(df['func']==func) & (df['net']==net) & (df['dataset']==dataset)]['test_acc'].item()
            value.append(test_acc)
    values.append(value)

cidx = pd.MultiIndex.from_arrays([
    [*[s for net in display_nets for s in itertools.repeat(net, len(display_funcs))]],
    [*(display_funcs*len(datasets))]
])

iidx = pd.MultiIndex.from_arrays([
    [*[s for dataset in display_datasets for s in itertools.repeat(dataset,len(display_funcs))]],
    [*(display_funcs*len(datasets))],
])

df = pd.DataFrame(
    data=values,columns=cidx,index=display_datasets
)

styler = df.style
styler = styler.format(na_rep='-',precision=2)

styler.to_latex(
    buf=os.path.join(pwd,'rgp_nodp.tex'),
    column_format='ccccccccc',
    hrules=True,
    multicol_align='c',
    clines='skip-last;data',
    convert_css=True,
)

