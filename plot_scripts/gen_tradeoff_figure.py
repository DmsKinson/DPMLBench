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
from name_map import NETS_MAP, DATASETS_MAP, FUNCS_CFG_MAP
from ploter import METRIC_DICT, MetricType

DEFAULT_CSV_MAP = {
    'epoch':0,'train_loss':1,'train_acc':2,'test_loss':3,'test_acc':4, 'train_cost':5, 'test_cost':6
}

COLOR_LIST = [
    '#FEF5B9','#FEE699','#FED36F','#FE9929','#F07A19','#DC5E0B','#C24702','#9D3603','#772A05'
]

FUNCS_CFG = list(FUNCS_CFG_MAP.values())

NO_LABEL_DP_FUNCS = [FUNCS_CFG_MAP[func] for func in [
    'adp_alloc','adpclip','gep','handcraft','loss','relu','tanh','rgp','knn','pate'
]]

DPSGD_FUNCS =[FUNCS_CFG_MAP[func] for func in [
    'relu','adp_alloc','adpclip','rgp','gep'
]] 

ML_FUNCS = [FUNCS_CFG_MAP[func] for func in [
    'handcraft','loss','tanh'
]]

DATA_PREPARATION_FUNCS = [FUNCS_CFG_MAP[func] for func in [
    'relu','handcraft',
]]

MODEL_DESIGN_FUNCS = [FUNCS_CFG_MAP[func] for func in [
    'relu','loss','tanh'
]]

MODEL_TRAINING_FUNCS = [FUNCS_CFG_MAP[func] for func in [
    'relu','adp_alloc','adpclip','rgp','gep'
]]

MODEL_ENSEMBLE_FUNCS = [FUNCS_CFG_MAP[func] for func in [
    'relu','knn','pate'
]]

LABELDP_COMPARASION_FUNCS = [FUNCS_CFG_MAP[func] for func in [
    'relu','tanh','rgp','knn','lp-2st','alibi'## + dp synthetic 
]]

FUNCS_DICT={
    'all': FUNCS_CFG,
    'nolabel': NO_LABEL_DP_FUNCS,
    'dpsgd':DPSGD_FUNCS,
    'ml':ML_FUNCS,
    'data_preparation':DATA_PREPARATION_FUNCS,
    'model_design':MODEL_DESIGN_FUNCS,
    'model_train':MODEL_TRAINING_FUNCS,
    'model_ensemble':MODEL_ENSEMBLE_FUNCS,
    'label_cmp':LABELDP_COMPARASION_FUNCS
}

DB_NETS = list(NETS_MAP.keys())

DB_DATASETS = list(DATASETS_MAP.keys())

# Temporal map for comparasion between simple and simplenn
SIMPLENETS_MAP = {
    'simple':'SimpleCNN',
    'simplenn':'SimpleCNN(small)'
}

class TradeOffPloter():
    def __init__(
            self,
            dpi=300,
            metric_type:str='auc',
            attack_type:str = 'black',
            shadow_dp:bool = False,
            **kwargs
        ) -> None:
        plt.style.use('ggplot')
        self.X_TICK_SIZE = 10
        self.Y_TICK_SIZE = 8
        self.SUB_PLOT_WIDTH = 4
        self.SUB_PLOT_HEIGHT = 2.5
        self.SUPLABEL_SIZE = 16
        self.LABEL_SIZE = 12
        self.DPI = dpi
        self.inf_eps = 10000
        self.FUNCS_CFG = kwargs.get('funcs_cfg',FUNCS_CFG)

        net_list = kwargs.get('nets',DB_NETS)
        dataset_list = kwargs.get('datasets',DB_DATASETS)
        self.format = kwargs.get('format','png')
        default_name = f'tradeoff_{attack_type}_{metric_type}_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}'
        self.figure_name = kwargs.get('figure_name')
        if(self.figure_name == None):
            self.figure_name = default_name

        self.shadow_dp = shadow_dp
        self.metric_type = METRIC_DICT[metric_type]
        self.attack_type = attack_type
        self.NETS_MAP = dict(zip(net_list,[NETS_MAP[net] for net in net_list]))
        self.DATASETS_MAP = dict(zip(dataset_list,[DATASETS_MAP[data] for data in dataset_list]))
        self.n_col = len(self.NETS_MAP)
        self.n_row = len(self.DATASETS_MAP)
        self.fig_width = self.SUB_PLOT_WIDTH * self.n_col;
        self.fig_height = self.SUB_PLOT_HEIGHT * self.n_row;
        self.eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
        self.xlabels = self.eps
        self.fig , self.axes_tuple = plt.subplots(self.n_row,self.n_col, figsize=(self.fig_width,self.fig_height),sharex='col',sharey='row',dpi=self.DPI,squeeze=False)
        self.fig.subplots_adjust(wspace=0.1, hspace=0.1)
        self.supxlabel = 'Utiltity Loss (%)'
        self.supylabel = 'Privacy Leakage (%)'


    def set_axe_format(self, x, net, y, dataset, axe:Axes):
        axe.xaxis.set_tick_params(labelsize=self.X_TICK_SIZE)
        axe.yaxis.set_tick_params(labelsize=self.Y_TICK_SIZE)
        axe.yaxis.set_label_position("right")
        axe.xaxis.set_label_position("top")
        if(y==0):
            axe.set_xlabel(net, fontsize=self.LABEL_SIZE)
        if(x==3):
            axe.set_ylabel(dataset, fontsize=self.LABEL_SIZE,rotation=90,rotation_mode='anchor',)

    def plot_and_save(self):
        for x, (net, display_net) in enumerate(self.NETS_MAP.items()) :
            for y, (dataset, display_dataset) in enumerate(self.DATASETS_MAP.items()) :
                axe:Axes = self.axes_tuple[y,x]
                self.set_axe_format(x, display_net, y, display_dataset, axe)
                # baseline = self.get_baseline(net, dataset)
                # axe.axhline(baseline, ls='--',c='black')
                for config in self.FUNCS_CFG:
                    # special case
                    if(config['db_name'] == 'gep' and net in ['inception','vgg']):
                        continue
                    ypoints = self.query_ypoints(config['db_name'], net, dataset,type=self.attack_type)
                    xs = self.query_xs(config['db_name'],net, dataset,)
                    assert len(xs)==len(ypoints), f"len(xs) and len(ypoints) is not equal:{len(xs)},{len(ypoints)}"
                    axe.plot(xs,ypoints,marker=config['marker'], markersize=4, c=config['color'],ls='--', label=config.get('display_name',config['db_name']))
                    axe.annotate(self.eps[0],xy=(xs[0],ypoints[0]))
                    axe.annotate(self.eps[-1],xy=(xs[-1],ypoints[-1]))
        # get legends from first setting, which has all funcs
        handles, labels = self.axes_tuple[0,0].get_legend_handles_labels()
        if(len(self.DATASETS_MAP)==1):
            self.fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5,-0.3),labelspacing=1,ncol=len(self.FUNCS_CFG))
            self.fig.supxlabel(self.supxlabel,y=-0.15 ,fontsize=self.SUPLABEL_SIZE)
        elif(len(self.DATASETS_MAP)==2):
            self.fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5,-0.1),labelspacing=1,ncol=len(self.FUNCS_CFG))
            self.fig.supxlabel(self.supxlabel,y=-0.02 ,fontsize=self.SUPLABEL_SIZE)
        else:
            self.fig.legend(handles, labels, loc='center',bbox_to_anchor=(0.5,0.03), ncol=len(self.FUNCS_CFG),labelspacing=1)
            self.fig.supxlabel(self.supxlabel, y=0.06, fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel(self.supylabel, x=0.09, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(f'figure/{self.figure_name}.{self.format}',bbox_inches='tight')

    def get_baseline_acc(self, net, dataset):
        ent = DB_Utility.get_or_none(
            DB_Utility.func=='relu', 
            DB_Utility.net==net, 
            DB_Utility.dataset==dataset, 
            DB_Utility.eps==None, 
        )
        if(ent == None):
            raise f"No baseline acc for {net},{dataset}"
        return ent.test_acc

    def get_baseline_mia(self, net, dataset, type):
        ent = DB_Privacy.get_or_none(
            DB_Privacy.func=='relu',
            DB_Privacy.net==net, 
            DB_Privacy.dataset==dataset, 
            DB_Privacy.eps==None, 
            DB_Privacy.type==type, 
            DB_Privacy.shadow_dp==False
        )
        if(ent == None):
            raise f"No baseline mia for {net},{dataset}"
        metric = self.get_metric(ent)
        print(f'baseine of {net}_{dataset} is ',metric)
        return metric

    def get_metric(self,ent):
        '''
        Get exact metric by specific type
        '''
        if(self.metric_type == MetricType.AUC):
            return max(ent.auc - 0.5,0)
        elif(self.metric_type == MetricType.PRECISION):
            return ent.precision
        elif(self.metric_type == MetricType.RECALL):
            return ent.recall
        elif(self.metric_type == MetricType.YEOM_LEAKAGE):
            return max(ent.ma,0)
        elif(self.metric_type == MetricType.F1):
            return ent.f1
        elif(self.metric_type == MetricType.ACC):
            return ent.asr
        else:
            raise NotImplementedError(f'Invalid metric type:{self.metric_type}')

    def query_ypoints(self, func, net, dataset, type):
        baseline = self.get_baseline_mia(net,dataset,type)

        ents = DB_Privacy.select().order_by(DB_Privacy.eps).where(
            DB_Privacy.func == func,
            DB_Privacy.net == net,
            DB_Privacy.dataset == dataset,
            DB_Privacy.eps << self.eps,
            DB_Privacy.type == type,
            DB_Privacy.shadow_dp == False,
            DB_Privacy.extra == None
        )
        xs = [100*self.get_metric(ent)/(baseline+1e-8) for ent in ents]
        return xs
        

    def query_xs(self, func, net, dataset):
        baseline = self.get_baseline_acc(net,dataset)
        ents = DB_Utility.select().order_by(DB_Utility.eps).where(
            DB_Utility.func == func,
            DB_Utility.net == net,
            DB_Utility.dataset == dataset,
            DB_Utility.eps << self.eps,
            DB_Utility.type == "target",
            DB_Utility.extra == None,
        )
        print(func,net,dataset)
        ypoints = [100*(1- ent.test_acc/baseline) for ent in ents]
        print(ypoints)
        return ypoints

def main(args):
    TradeOffPloter(
        metric_type=args.metric,
        attack_type=args.type,
        shadow_dp=args.shadow_dp,
        funcs_cfg=FUNCS_DICT[args.funcs],
        nets=args.nets,
        datasets=args.datasets,
        figure_name = args.name,
        format=args.format
    ).plot_and_save()

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--shadow_dp', action='store_true')
    parser.add_argument('--nets', type=str, nargs='+', help='nets list to plot', choices=['simplenn','resnet','inception','vgg'])
    parser.add_argument('--datasets', type=str, nargs='+', help='datasets list to plot',choices=['mnist','fmnist','svhn','cifar10'])
    parser.add_argument('--type', type=str, default='black',choices=['black','white','label'])
    parser.add_argument('--metric', type=str, choices=['auc','precision','recall','yeom','ba','f1','acc'], default='auc')
    parser.add_argument('--funcs', type=str, default='all', choices=list(FUNCS_DICT.keys()))
    parser.add_argument('--format', type=str, default='png', choices=['png','pdf','jpg'])
    parser.add_argument('--name',type=str, help='file name without extensive name')
    args = parser.parse_args()
    if(args.nets==None):
        args.nets = DB_NETS
    if(args.datasets == None):
        args.datasets = DB_DATASETS
    main(args)