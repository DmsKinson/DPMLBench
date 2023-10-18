import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import pandas as pd
import itertools
import numpy as np
from db_models import DB_AccStat
from name_map import * 
import matplotlib.pyplot as plt

nets = ['simplenn','simplenn_norm']
datasets = ['fmnist','cifar10']
funcs = ['relu','tanh']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100]

COLOR_MAP = {
    'DP-SGD(ReLU)':'#9467BD',
    'DP-SGD(Tanh)':'#8C564B'
}
LINE_MAP = {
    'w/o GroupNorm':'-',
    'with GroupNorm':'--'
}

display_datasets = [DATASETS_MAP[dataset] for dataset in datasets]
display_nets = ['w/o GroupNorm','with GroupNorm']
display_eps = [str(e) for e in eps]
display_funcs = ['DP-SGD(ReLU)','DP-SGD(Tanh)']

ents = DB_AccStat.select().where(
    DB_AccStat.func << funcs,
    DB_AccStat.net << nets,

)

# DEFAULT_COL = ['epoch','train_loss','train_acc','test_loss','test_acc','train_cost','test_cost']
values = []
for ent in ents:
   
    value = [
        ent.func,
        ent.net,
        ent.dataset,
        ent.eps,
        ent.mean,
        ent.std
    ]
    values.append(value)

mean_df = pd.DataFrame(data=values,columns=['func','net','dataset','eps','mean','std'])

mean_list = []
std_list = []
for func in funcs:
    for net in nets:
        tmp_mean_list = []
        tmp_std_list = []
        for dataset in datasets:
            for e in eps:
                # print(net,dataset,e)
                # print(df[(df['func']=='relu') & (df['net']==net) & (df['dataset']==dataset) & (df['eps']==e)]['test_acc'])
                # assert False
                try:
                    mean = mean_df[(mean_df['func']==func) & (mean_df['net']==net) & (mean_df['dataset']==dataset) & (mean_df['eps']==e)]['mean'].item()
                    std = mean_df[(mean_df['func']==func) & (mean_df['net']==net) & (mean_df['dataset']==dataset) & (mean_df['eps']==e)]['std'].item()
                except:
                    print(func,net,dataset,e)
                    # print(df[(df['func']==func) & (df['net']==net) & (df['dataset']==dataset) & (df['eps']==e)]['test_acc'])
                    assert False
                tmp_mean_list.append(mean)
                tmp_std_list.append(std)
        mean_list.append(tmp_mean_list)
        std_list.append(tmp_std_list)


cidx = pd.MultiIndex.from_arrays([
    [*[s for dataset in display_datasets for s in itertools.repeat(dataset,len(eps))]],
    [*(display_eps*len(datasets))],
])

iidx = pd.MultiIndex.from_arrays([
    [*[s for func in display_funcs for s in itertools.repeat(func,len(funcs))]],
    [*(display_nets*len(funcs))],
])

mean_df = pd.DataFrame(
    data=mean_list,columns=cidx,index=iidx
)

std_df = pd.DataFrame(
    data=std_list,columns=cidx,index=iidx
)

print(mean_df)

plt.style.use('ggplot')
fig,axes = plt.subplots(1,2,figsize=(2*7,3.5),dpi=300)
fig.subplots_adjust(wspace=0.1, hspace=0.1)

markers = [['s','P'],['D','v']]
marker_size = 10
font_size = 18
for axe,dataset in zip(axes,display_datasets):
    axe.set_xscale('log')
    axe.yaxis.set_label_position("right")
    axe.xaxis.set_label_position("top")
    axe.set_xlabel(dataset,fontsize=font_size)
    for i,func in enumerate(display_funcs):
        for j,net in enumerate(display_nets):
            ys = mean_df[dataset].T[func,net].to_numpy()
            stds = std_df[dataset].T[func,net].to_numpy()
            axe.plot(eps,ys,label=f'{func} {net}',c=COLOR_MAP[func],ls=LINE_MAP[net],marker=markers[i][j],markersize=marker_size,markerfacecolor='white')
            # axe.errorbar(self.eps[:length], ypoints, stds, c=config['color'],fmt='none',capsize=3)

            axe.errorbar(eps,ys, stds, fmt='none',capsize=3,c=COLOR_MAP[func])
fig.supylabel('Accuracy (%)',x=0.08,fontsize=font_size)
# fig.supxlabel('Privacy Budget')
handles,labels = axe.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=font_size,bbox_to_anchor=(0.5,-0.3),labelspacing=1,ncol=len(funcs))
plt.savefig(os.path.join(pwd,'..','figure','simplenn_norm.pdf'),bbox_inches='tight')