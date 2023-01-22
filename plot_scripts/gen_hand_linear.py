import sys,os
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd,'..')) 

import argparse
from enum import Enum
from matplotlib import ticker
from matplotlib.axes import Axes
from db_models import DB_Attack, DB_Func, DB_Csv, DB_Model, DB_Privacy, DB_Utility
import matplotlib.pyplot as plt
import csv
import time
import numpy as np
from name_map import NETS_MAP, DATASETS_MAP, FUNCS_CFG_MAP,FUNC_DB2FIGURE,FUNC_DB2TABLE



X_TICK_SIZE = 10
Y_TICK_SIZE = 8
SUB_PLOT_WIDTH = 4.7
SUB_PLOT_HEIGHT = 2.5
SUPLABEL_SIZE = 16
LABEL_SIZE = 12
DPI = 300
inf_eps = 10000
    

eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
nets = ['linear','resnet']
datasets = ['mnist','fmnist','svhn','cifar10']
colors = ['#2CA02C','#B153C5']
markers = ['*','x']
name = 'hand_linear.png'

plt.style.use('ggplot')
fig, axes = plt.subplots(nrows=1,ncols=len(datasets),figsize=(SUB_PLOT_WIDTH*len(datasets),SUB_PLOT_HEIGHT),sharey='col',dpi=DPI,squeeze=True)
fig.subplots_adjust(wspace=0.1, hspace=0.1)

for net,color,marker in zip(nets,colors,markers):
    for dataset,axe in zip(datasets,axes):
        axe.set_xscale('log')
        ent = DB_Utility.select(
            DB_Utility.test_acc
        ).where(
            DB_Utility.func == 'handcraft',
            DB_Utility.net == net,
            DB_Utility.dataset == dataset,
            DB_Utility.eps << eps
        ).order_by(
            DB_Utility.eps
        )
        assert ent != None, f'No results for {net},{dataset}'
        ypoints = [e.test_acc for e in ent]
        axe.plot(eps,ypoints,c=color,label=net,marker=marker)
handles,labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,loc='lower center',bbox_to_anchor=(0.5,-0.2),labelspacing=1, ncol=len(nets),)
fig.savefig(os.path.join(pwd,name),bbox_inches='tight')

