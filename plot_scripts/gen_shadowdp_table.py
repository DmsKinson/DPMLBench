import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import pandas as pd
from db_models import DB_Utility
from name_map import *

nets = ['simplenn','resnet','inception','vgg']
datasets = ['mnist','fmnist','svhn','cifar10']

display_eps = ['0.2','0,3','0.4','0.5','1','2','4','8','100','1000']
display_nets = [NETS_MAP[n] for n in nets]


values = []

for idx,net in enumerate(nets) :
    acc_ents = DB_Utility.select().order_by(DB_Utility.eps).where(
        DB_Utility.func=='handcraft',
        DB_Utility.dataset == dataset,
        DB_Utility.net == net,
        DB_Utility.type=='target',
        # DB_Utility.eps << display_eps
    )
    ys = [ent.test_acc for ent in acc_ents]
    assert len(ys)==11, f'Illegal length of query results:{len(ys)}'
    ys = [ys[1],ys[3],ys[5],ys[7],ys[9],ys[10],ys[0]]
    values.append(ys)

print(display_nets)
print(values)
iidx = pd.MultiIndex.from_arrays([
    [*[net for net in display_nets ]],
])

df = pd.DataFrame(
    data=values,columns=display_eps,index=iidx
)

styler = df.style
styler = styler.format(na_rep='-',precision=2)

styler.to_latex(
    buf=os.path.join(pwd,'acc_hand_linear.tex'),
    column_format=('ccccccc'),
    hrules=True,
    multicol_align='|c|',
    clines='skip-last;data',
    convert_css=True,
)
