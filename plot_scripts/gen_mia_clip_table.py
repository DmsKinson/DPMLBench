import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import pandas as pd
from db_models import DB_Privacy, DB_Utility, db
import itertools
import numpy as np
from name_map import *
import matplotlib.pyplot as plt 

plt.style.use('ggplot')
nets = ['simplenn','resnet','inception','vgg']
datasets = ['cifar10']

display_eps = [8,100,1000,'inf(clip)','inf',]
display_nets = [NETS_MAP[n] for n in nets]
display_datasets = [DATASETS_MAP[d] for d in datasets]

values = []
for idx,net in enumerate(nets) :
    mia_ents = DB_Privacy.select().order_by(DB_Privacy.extra,DB_Privacy.eps).where(
        DB_Privacy.func=='relu',
        DB_Privacy.dataset == 'cifar10',
        DB_Privacy.net == net,
        DB_Privacy.type=='black',
        DB_Privacy.shadow_dp==False,
    )
    acc_ents = DB_Utility.select().order_by(DB_Utility.extra,DB_Utility.eps).where(
        DB_Utility.func=='relu',
        DB_Utility.dataset == 'cifar10',
        DB_Utility.net == net,
        DB_Utility.type=='target',
    )
    ys = [ent.test_acc for ent in acc_ents]
    ys = [*ys[8:],ys[0]]
    values.append(ys)
    ys = [max(ent.auc,0) for ent in mia_ents]
    ys = [*ys[8:],ys[0]]
    values.append(ys)
    

iidx = pd.MultiIndex.from_arrays([
    [*[s for net in display_nets for s in itertools.repeat(net,2)]],
    [*(['ACC','AUC']*len(nets))],
])

df = pd.DataFrame(
    data=values,columns=display_eps,index=iidx
)

styler = df.style
styler = styler.format(na_rep='-',precision=2)

styler.to_latex(
    buf=os.path.join(pwd,'mia_clip.tex'),
    column_format=('ccccccc'),
    hrules=True,
    multicol_align='|c|',
    clines='skip-last;data',
    convert_css=True,
)
