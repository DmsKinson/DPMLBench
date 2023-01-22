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
    'relu','handcraft','dpgen','gswgan'
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
    'relu','tanh','rgp','knn','lp-2st','alibi','dpgen' 
]]

PATE_COMPARE_FUNCS = [
    {
        'db_name':'knn',
        'color':'#AAD8B3',
        'display_name':'KNN_UDA',
        'table_name':FUNC_DB2TABLE['knn'],
        'marker':'+',
        'extra':'uda',
    },
    {
        'db_name':'knn',
        'color':'#72D884',
        'display_name':FUNC_DB2FIGURE['knn'],
        'table_name':FUNC_DB2TABLE['knn'],
        'marker':'+',
    },
    {
        'db_name':'pate',
        'color':'#B7B7B7',
        'display_name':'PATE_UDA',
        'table_name':FUNC_DB2TABLE['pate'],
        'marker':'p',
        'extra':'uda'
    },
    {
        'db_name':'pate',
        'color':'#7f7f7f',
        'display_name':FUNC_DB2FIGURE['pate'],
        'table_name':FUNC_DB2TABLE['pate'],
        'marker':'p',
    },
]

FUNCS_DICT={
    'all': FUNCS_CFG,
    'nolabel': NO_LABEL_DP_FUNCS,
    'dpsgd':DPSGD_FUNCS,
    'ml':ML_FUNCS,
    'data_preparation':DATA_PREPARATION_FUNCS,
    'model_design':MODEL_DESIGN_FUNCS,
    'model_train':MODEL_TRAINING_FUNCS,
    'model_ensemble':MODEL_ENSEMBLE_FUNCS,
    'label_cmp':LABELDP_COMPARASION_FUNCS,
    'pate_cmp':PATE_COMPARE_FUNCS
}

DB_NETS = list(NETS_MAP.keys())

DB_DATASETS = list(DATASETS_MAP.keys())

# Temporal map for comparasion between simple and simplenn
SIMPLENETS_MAP = {
    'simple':'SimpleCNN',
    'simplenn':'SimpleCNN(small)'
}

class MetricType(Enum):
        AUC = 0
        PRECISION = 1
        RECALL = 2
        YEOM_LEAKAGE = 3
        BA = 4
        F1 = 5
        ACC = 6
        AUC_NORM = 7
        AUC_TAILOR = 8

METRIC_DICT={
    'auc':MetricType.AUC,
    'precision':MetricType.PRECISION,
    'recall':MetricType.RECALL,
    'yeom':MetricType.YEOM_LEAKAGE,
    'ba':MetricType.BA,
    'f1':MetricType.F1,
    'acc':MetricType.ACC,
    'auc_norm':MetricType.AUC_NORM,
    'auc_tailored':MetricType.AUC_TAILOR
}

class AccPloter():
    def __init__(self,dpi=300, **kwargs) -> None:
        plt.style.use('ggplot')
        self.X_TICK_SIZE = 10
        self.Y_TICK_SIZE = 8
        self.SUB_PLOT_WIDTH = 4.7
        self.SUB_PLOT_HEIGHT = 2.5
        self.SUPLABEL_SIZE = 16
        self.LABEL_SIZE = 12
        self.DPI = dpi
        self.inf_eps = 10000
        self.FUNCS_CFG = kwargs.get('funcs_cfg',FUNCS_CFG)
        net_list = kwargs.get('nets',DB_NETS)
        dataset_list = kwargs.get('datasets',DB_DATASETS)
        self.format = kwargs.get('format','png')
        default_name = f'acc_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}'
        self.figure_name = kwargs.get('figure_name')
        if(self.figure_name == None):
            self.figure_name = default_name
        self.NETS_MAP = dict(zip(net_list,[NETS_MAP[net] for net in net_list]))
        self.DATASETS_MAP = dict(zip(dataset_list,[DATASETS_MAP[net] for net in dataset_list]))
        self.n_col = len(self.NETS_MAP)
        self.n_row = len(self.DATASETS_MAP)
        self.fig_width = self.SUB_PLOT_WIDTH * self.n_col;
        self.fig_height = self.SUB_PLOT_HEIGHT * self.n_row;
        self.eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
        self.xlabels = self.eps
        self.fig , self.axes_tuple = plt.subplots(self.n_row,self.n_col, figsize=(self.fig_width,self.fig_height),sharex='col',sharey='row',dpi=self.DPI,squeeze=False)
        self.fig.subplots_adjust(wspace=0.1, hspace=0.1)

    def set_axe_format(self, x, net, y, dataset, axe:Axes):
        axe.set_xscale('log')
        axe.xaxis.set_tick_params(labelsize=self.X_TICK_SIZE)
        axe.yaxis.set_tick_params(labelsize=self.Y_TICK_SIZE)
        axe.yaxis.set_label_position("right")
        axe.xaxis.set_label_position("top")
        axe.set_ylim(-1,101)
        axe.set_xlim(0.15,1002)
        # axe.set_xticks(ticks=self.eps)
        # axe.set_xticklabels(self.xlabels, rotation=270, ha="center",va='center_baseline')
        # axe.xaxis.set_major_formatter(ticker.ScalarFormatter())
        # axe.xaxis.set_minor_locator(ticker.NullLocator())
        # axe.xaxis.set_major_locator(ticker.FixedLocator(self.eps))
        # axe.xaxis.set_major_formatter(ticker.FixedFormatter(self.xlabels))
        # axe.yaxis.set_major_locator(ticker.FixedLocator([0,20,40,60,80,100]))
        if(y==0):
            axe.set_xlabel(net, fontsize=self.LABEL_SIZE)
        if(x==3):
            axe.set_ylabel(dataset, fontsize=self.LABEL_SIZE,rotation=90,rotation_mode='anchor',)

    def plot_and_save(self):
        for x, (net, display_net) in enumerate(self.NETS_MAP.items()) :
            for y, (dataset, display_dataset) in enumerate(self.DATASETS_MAP.items()) :
                axe:Axes = self.axes_tuple[y,x]
                self.set_axe_format(x, display_net, y, display_dataset, axe)
                baseline = self.get_baseline(net, dataset)
                axe.axhline(baseline, ls='--',c='black')
                for config in self.FUNCS_CFG:
                    # special case
                    if(config['db_name'] == 'gep' and net in ['inception','vgg']):
                        continue
                    ypoints = self.query_ypoints(config['db_name'], net, dataset, config)
                    # ypoints = [100*(1-y/baseline) for y in ypoints]
                    length = len(ypoints)
                    axe.plot(self.eps[:length],ypoints,marker=config['marker'], markersize=4, c=config['color'], label=config.get('display_name',config['db_name']))
        # get legends from simple-mnist setting, which has all funcs
        handles, labels = self.axes_tuple[0,0].get_legend_handles_labels()
        if(len(self.DATASETS_MAP)==1):
            self.fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5,-0.3),labelspacing=1,ncol=len(self.FUNCS_CFG))
            self.fig.supxlabel('Privacy Budget',y=-0.15 ,fontsize=self.SUPLABEL_SIZE)
        elif(len(self.DATASETS_MAP)==2):
            self.fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5,-0.1),labelspacing=1,ncol=len(self.FUNCS_CFG))
            self.fig.supxlabel('Privacy Budget',y=-0.02 ,fontsize=self.SUPLABEL_SIZE)
        else:
            self.fig.legend(handles, labels, loc='center',bbox_to_anchor=(0.5,0.03), ncol=len(self.FUNCS_CFG),labelspacing=1)
            self.fig.supxlabel('Privacy Budget', y=0.06, fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel('Accuracy (%)', x=0.09, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(f'{pwd}/../figure/{self.figure_name}.{self.format}',bbox_inches='tight')

    def get_baseline(self, net, dataset):
        ent = DB_Utility.get_or_none(
            DB_Utility.func=='relu', 
            DB_Utility.net==net, 
            DB_Utility.dataset==dataset, 
            DB_Utility.eps==None, 
        )
        if(ent == None):
            print(f'==> No baseline for {net}_{dataset}')
            return 100
        return ent.test_acc

    def query_ypoints(self, func, net, dataset, config):
        ypoints = []
        for eps in self.eps:
            # convert inf_eps to None in db query
            if(eps == self.inf_eps):
                eps = None
            ent = DB_Utility.get_or_none(
                DB_Utility.func==func, 
                DB_Utility.net==net, 
                DB_Utility.dataset==dataset, 
                DB_Utility.eps==eps,
                DB_Utility.type=='target',
                DB_Utility.extra==config.get('extra',None)
            )
            # TODO: remove when all experiments end
            if(ent == None):
                if(eps == None):
                    continue
                ypoints.append(0)
                print(f'==> No record for {func}_{net}_{dataset}_{eps}')
                continue
            try:
                ypoints.append(ent.test_acc) 
            except Exception as e:
                print('Error:',e)
                ypoints.append(0)
        return np.array(ypoints) 

class MiaPloter(AccPloter):
    def __init__(
        self,
        dpi=300,
        metric_type:str='auc_norm',
        attack_type:str = 'black',
        shadow_dp:bool = False,
        **kwargs
    ) -> None:
        super().__init__(dpi,**kwargs)
        plt.style.use('ggplot')
        self.EPS = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
        self.bar_width = 0.4
        self.group_margin = 0.3
        self.shadow_dp = shadow_dp

        default_name = f'mia_{attack_type}_{metric_type}_{shadow_dp}_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}'
        self.figure_name = kwargs.get('figure_name')
        if(self.figure_name == None):
            self.figure_name = default_name
        self.n_group = len(self.EPS)
        self.n_label_per_group = len(self.FUNCS_CFG)
        self.group_width = self.n_label_per_group*self.bar_width
        self.group_l = np.array([self.group_margin*2 + i*(self.group_width + 2*self.group_margin) for i in range(self.n_group)])
        self.metric_type = METRIC_DICT[metric_type]
        self.attack_type = attack_type
        self.supxlabel = 'Privacy Budget'
        self.supylabel = 'Privacy Leakage (%)'

    def plot_and_save(self):
        for x, (net, display_net) in enumerate(self.NETS_MAP.items()) :
            for y, (dataset, display_dataset) in enumerate(self.DATASETS_MAP.items()) :
                axe:Axes = self.axes_tuple[y,x]
                self.set_axe_format(x, display_net, y, display_dataset, axe)
                
                baseline = self.get_baseline(net, dataset,self.attack_type)
                # axe.axhline(baseline, ls='--',c='black')

                for i_func, config in enumerate(self.FUNCS_CFG) :
                    # special case
                    if(config['db_name'] == 'gep' and net in ['inception','vgg']):
                        continue
                    ypoints = self.query_ypoints(config['db_name'], net, dataset, self.shadow_dp)
                    ## propotion to baseline
                    ypoints = [100*y/(baseline+1e-8) for y in ypoints]
                    axe.bar(self.group_l+i_func*self.bar_width, ypoints, width=self.bar_width, color=config['color'], edgecolor='white', linewidth=0, label=config.get('display_name',config['db_name']),align='edge')

        handles, labels = self.axes_tuple[0,0].get_legend_handles_labels()
        if(len(self.DATASETS_MAP)==1):
            self.fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5,-0.26),labelspacing=1,ncol=len(self.FUNCS_CFG))
            self.fig.supxlabel(self.supxlabel,y=-0.1 ,fontsize=self.SUPLABEL_SIZE)
        elif(len(self.DATASETS_MAP)==2):
            self.fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5,-0.1),labelspacing=1,ncol=len(self.FUNCS_CFG))
            self.fig.supxlabel(self.supxlabel,y=-0.02 ,fontsize=self.SUPLABEL_SIZE)
        else:
            self.fig.legend(handles, labels, loc='center',bbox_to_anchor=(0.5,0.03), ncol=len(self.FUNCS_CFG),labelspacing=1)
            self.fig.supxlabel(self.supxlabel, y=0.06, fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel(self.supylabel, x=0.09, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(f'{pwd}/../{self.figure_name}.{self.format}',bbox_inches='tight')
        
    def get_metric(self,ent):
        '''
        Get exact metric by specific type
        '''
        if(self.metric_type == MetricType.AUC_TAILOR):
            return max(ent.auc, 0.5)
        if(self.metric_type == MetricType.AUC):
            return ent.auc
        if(self.metric_type == MetricType.AUC_NORM):
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
            raise NotImplementedError('Invalid metric type')

    def query_ypoints(self, func, net, dataset, shadow_dp=False):
        ypoints = []
        for eps in self.EPS:
            if(eps == 'inf'):
                eps = None
            ent = DB_Privacy.get_or_none(
                DB_Privacy.func==func, 
                DB_Privacy.net==net,
                DB_Privacy.dataset==dataset,
                DB_Privacy.type==self.attack_type, 
                DB_Privacy.eps==eps, 
                DB_Privacy.shadow_dp==shadow_dp
            )
            if(ent == None):
                ypoints.append(0)
                print(f'==> No record for {self.attack_type}_{func}_{net}_{dataset}_{eps}')
                continue
            metric = self.get_metric(ent)
            ypoints.append(metric)
        return ypoints

    def get_baseline(self, net, dataset, type):
        ent = DB_Privacy.get_or_none(
            DB_Privacy.func=='relu',
            DB_Privacy.net==net, 
            DB_Privacy.dataset==dataset, 
            DB_Privacy.eps==None, 
            DB_Privacy.type==type, 
            DB_Privacy.shadow_dp==False
        )
        if(ent == None):
            print(f'==> No baseline for {net}_{dataset}')
            return 0
        metric = self.get_metric(ent)
        print(f'baseine of {net}_{dataset} is ',metric)
        return metric

    def set_axe_format(self, x, net, y, dataset, axe):
        axe.yaxis.set_label_position("right")
        axe.xaxis.set_label_position("top")
        if(self.metric_type == MetricType.YEOM_LEAKAGE):
            axe.set_ylim(0,100)
        else:
            axe.set_ylim(0,100)
        axe.set_xlim(0, self.group_l[-1]+self.group_margin*2+self.group_width)
        axe.tick_params(axis='x', which='both', length=1, width=2)
        # l,r = axe.get_xlim()
        axe.set_xticks(ticks=self.group_l+self.group_width/2 )
        axe.set_xticklabels(labels=self.EPS)
        
        if(y==0):
            axe.set_xlabel(net, fontsize=self.LABEL_SIZE)            
        if(x==3):
            axe.set_ylabel(dataset, fontsize=self.LABEL_SIZE,rotation=90,rotation_mode='anchor',)

        axe.xaxis.set_tick_params(labelsize=self.X_TICK_SIZE)
        axe.yaxis.set_tick_params(labelsize=self.Y_TICK_SIZE)

class MultiMiaPloter(MiaPloter):
    def __init__(self, dpi=300, metric_type: str = 'auc', attack_type: str = 'black', shadow_dp: bool = False, **kwargs) -> None:
        super().__init__(dpi, metric_type, attack_type, shadow_dp, **kwargs)
        self.fig_width = 18
        self.fig_height = 4
        default_name = f'multimia_{metric_type}_{shadow_dp}_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}'
        self.figure_name = kwargs.get('figure_name')
        if(self.figure_name == None):
            self.figure_name = default_name
        self.attack_types = ['black','white','label']
        self.dataset = kwargs.get('dataset','cifar10')
        self.display_dataset = 'CIFAR-10'
        self.fig , self.axes_tuple = plt.subplots(nrows=len(self.attack_types),ncols=len(self.NETS_MAP), figsize=(self.fig_width,self.fig_height),sharex='col',sharey='row', dpi=self.DPI,squeeze=False)
        self.fig.subplots_adjust(wspace=0.05, hspace=0)

    def get_baseline(self, net, dataset, attack_type):
        ent = DB_Privacy.get_or_none(
            DB_Privacy.func=='relu',
            DB_Privacy.net==net, 
            DB_Privacy.dataset==dataset, 
            DB_Privacy.eps==None, 
            DB_Privacy.type==attack_type, 
            DB_Privacy.shadow_dp==False
        )
        if(ent == None):
            print(f'==> No baseline for {net}_{dataset}')
            return 0
        metric = self.get_metric(ent)
        print(f'baseine of {net}_{dataset} is ',metric)
        return metric

    def set_axe_format(self, x, net, y, dataset, axe):
        axe.yaxis.set_label_position("right")
        axe.xaxis.set_label_position("top")
        
        # axe.set_ylim(0,35)
        axe.set_xlim(0, self.group_l[-1]+self.group_margin*2+self.group_width)
        axe.tick_params(axis='x', which='both', length=1, width=2)
        # l,r = axe.get_xlim()
        axe.set_xticks(ticks=self.group_l+self.group_width/2 )
        axe.set_xticklabels(labels=self.EPS)
        
        if(y==0):
            axe.set_xlabel(net, fontsize=self.LABEL_SIZE)            
        if(x==3):
            axe.set_ylabel(dataset, fontsize=self.LABEL_SIZE,rotation=90,rotation_mode='anchor',)

        axe.xaxis.set_tick_params(labelsize=self.X_TICK_SIZE)
        axe.yaxis.set_tick_params(labelsize=self.Y_TICK_SIZE)

    def query_ypoints(self, func, net, dataset, attack_type, config,shadow_dp=False):
        ypoints = []
        for eps in self.EPS:
            if(eps == 'inf'):
                eps = None
            ent = DB_Privacy.get_or_none(
                DB_Privacy.func==func, 
                DB_Privacy.net==net,
                DB_Privacy.dataset==dataset,
                DB_Privacy.type==attack_type, 
                DB_Privacy.eps==eps, 
                DB_Privacy.shadow_dp==shadow_dp,
                DB_Privacy.extra == config.get('extra',None),
            )
            if(ent == None):
                ypoints.append(0)
                print(f'==> No record for {attack_type}_{func}_{net}_{dataset}_{eps}')
                continue
            metric = self.get_metric(ent)
            ypoints.append(metric)
        return ypoints

    def plot_and_save(self):
        for x, (net, display_net) in enumerate(self.NETS_MAP.items()) :
            for y, attack_type in enumerate(self.attack_types) :
                axe:Axes = self.axes_tuple[y,x]
                self.set_axe_format(x, display_net, y, attack_type.capitalize(), axe)
                
                baseline = self.get_baseline(net, self.dataset, attack_type,)
                # axe.axhline(baseline, ls='--',c='black')

                for i_func, config in enumerate(self.FUNCS_CFG) :
                    # special case
                    if(config['db_name'] == 'gep' and net in ['inception','vgg']):
                        continue
                    ypoints = self.query_ypoints(config['db_name'], net, self.dataset, attack_type, config, self.shadow_dp)
                    ## propotion to baseline (metric should be auc_norm)
                    ypoints = [100*y/(baseline+1e-8) for y in ypoints]
                    # ypoints = [baseline-y for y in ypoints]
                    axe.bar(self.group_l+i_func*self.bar_width, ypoints, width=self.bar_width, color=config['color'], edgecolor='white', linewidth=0, label=config.get('display_name',config['db_name']),align='edge')
        handles, labels = self.axes_tuple[0,0].get_legend_handles_labels()
        
        self.fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5,-0.15),labelspacing=1,ncol=len(self.FUNCS_CFG))
        self.fig.supxlabel(self.supxlabel,y=-0.04 ,fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel(self.supylabel, x=0.09, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(f'{pwd}/../figure/{self.figure_name}.{self.format}',bbox_inches='tight')

class ShadowDPPloter(MiaPloter):
    def __init__(self, dpi=300, metric_type: str = 'auc', attack_type: str = 'black', shadow_dp: bool = False, **kwargs) -> None:
        super().__init__(dpi, metric_type, attack_type, shadow_dp, **kwargs)        
        self.name = f'mia_test_shadow_{metric_type}_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}.{self.format}'
        self.bar_width = 0.4
        self.supylabel = "Tailored AUC"

    def set_axe_format(self, x, net, y, dataset, axe):
        axe.yaxis.set_label_position("right")
        axe.xaxis.set_label_position("top")
        
        axe.set_ylim(0,1)
        axe.set_xlim(0, self.group_l[-1]+self.group_margin*2+self.group_width)
        axe.tick_params(axis='x', which='both', length=1, width=2)
        # l,r = axe.get_xlim()
        axe.set_xticks(ticks=self.group_l+self.group_width/2 )
        axe.set_xticklabels(labels=self.EPS)
        
        if(y==0):
            axe.set_xlabel(net, fontsize=self.LABEL_SIZE)            
        if(x==3):
            axe.set_ylabel(dataset, fontsize=self.LABEL_SIZE,rotation=90,rotation_mode='anchor',)

        axe.xaxis.set_tick_params(labelsize=self.X_TICK_SIZE)
        axe.yaxis.set_tick_params(labelsize=self.Y_TICK_SIZE)

    def plot_and_save(self):
        for x, (net, display_net) in enumerate(self.NETS_MAP.items()) :
            for y, (dataset, display_dataset) in enumerate(self.DATASETS_MAP.items()) :
                axe:Axes = self.axes_tuple[y,x]
                self.set_axe_format(x, display_net, y, display_dataset, axe)
                
                # baseline = self.get_baseline(net, dataset,'white_old')
                axe.axhline(0.5, ls='--',c='#AB4533')

                # baseline = self.get_baseline(net, dataset,'white')
                # axe.axhline(baseline, ls='--',c='#FF7F0E')

                for i_func, config in enumerate(self.FUNCS_CFG) :
                    ypoints = self.query_ypoints(config['db_name'], net, dataset, config['shadow_dp'])
                    axe.bar(self.group_l+i_func*self.bar_width, ypoints, width=self.bar_width, color=config['color'], label=config.get('display_name',config['db_name']),align='edge')
        
        handles, labels = axe.get_legend_handles_labels()
        self.fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(0.5,-0.25),labelspacing=1,ncol=len(self.FUNCS_CFG))
        self.fig.supxlabel(self.supxlabel,y=-0.08 ,fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel(self.supylabel, x=0.09, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(f'figure/{self.name}',bbox_inches='tight')


def main(args):
    # baseline 
    if(args.acc):
        print('plot acc')
        AccPloter(
            nets=args.nets,
            datasets=args.datasets,
            figure_name = args.name,
            funcs_cfg=FUNCS_DICT[args.funcs],
            format=args.format
        ).plot_and_save()

    if(args.mia):
        print('plot mia')
        MiaPloter(
            metric_type=args.metric,
            attack_type=args.type,
            shadow_dp=args.shadow_dp,
            funcs_cfg=FUNCS_DICT[args.funcs],
            nets=args.nets,
            datasets=args.datasets,
            figure_name = args.name,
            format=args.format
        ).plot_and_save()
    # comparison between old and new attack network architecture
    if(args.test):
        print('test')
        TEMP_FUNCS_CFG = [
            {
                'db_name':'relu',
                'type':'black',
                'color':'#FF7F0E',
                'display_name':'shadow with dp',
                'shadow_dp':True,
                'marker':'>'
            },
            {
                'db_name':'relu',
                'type':'black',
                'color':'#2CA02C',
                'display_name':'shadow w/o dp',
                'shadow_dp':False,
                'marker':'<'
            },
        ]
        ShadowDPPloter(
            metric_type=args.metric,
            attack_type=args.type,
            shadow_dp=args.shadow_dp,
            funcs_cfg=TEMP_FUNCS_CFG,
            nets=args.nets,
            datasets=args.datasets,
            figure_name = args.name,
            format=args.format
        ).plot_and_save()

    if(args.multi):
        MultiMiaPloter(
            metric_type=args.metric,
            attack_type=args.type,
            shadow_dp=args.shadow_dp,
            funcs_cfg=FUNCS_DICT[args.funcs],
            nets=args.nets,
            dataset = args.dataset,
            figure_name = args.name,
            format=args.format
        ).plot_and_save()

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--acc','-a', action='store_true')
    parser.add_argument('--mia','-m', action='store_true')
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--shadow_dp', action='store_true')

    parser.add_argument('--nets', type=str, nargs='+', help='nets list to plot', choices=['simplenn','resnet','inception','vgg'])
    parser.add_argument('--datasets', type=str, nargs='+', help='datasets list to plot',choices=['mnist','fmnist','svhn','cifar10'])
    parser.add_argument('--dataset',type=str,choices=['mnist','fmnist','svhn','cifar10'],help='specific dataset for multi-attack ploter')
    parser.add_argument('--type', type=str, default='black',choices=['black','white','label'])
    parser.add_argument('--metric', type=str, choices=['auc','precision','recall','yeom','ba','f1','acc','auc_norm'], default='auc_norm')
    parser.add_argument('--funcs', type=str, default='all', choices=list(FUNCS_DICT.keys()))
    parser.add_argument('--format', type=str, default='png', choices=['png','pdf','jpg'])
    parser.add_argument('--name',type=str, help='file name without extensive name')
    args = parser.parse_args()
    if(args.nets==None):
        args.nets = DB_NETS
    if(args.datasets == None):
        args.datasets = DB_DATASETS
    main(args)







