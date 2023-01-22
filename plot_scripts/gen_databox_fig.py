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
from ploter import FUNCS_CFG,DB_DATASETS,DB_NETS,METRIC_DICT,MetricType,FUNCS_DICT

class NetBoxPloter():
    def __init__(self, dpi=300,**kwargs) -> None:
        plt.style.use('ggplot')
        self.X_TICK_SIZE = 10
        self.Y_TICK_SIZE = 8
        self.SUB_PLOT_WIDTH = 20
        self.SUB_PLOT_HEIGHT = 4
        self.SUPLABEL_SIZE = 16
        self.LABEL_SIZE = 12
        self.DPI = dpi
        self.fig , self.axes_tuple = plt.subplots(1,4, figsize=(self.SUB_PLOT_WIDTH,self.SUB_PLOT_HEIGHT),sharey='row',dpi=self.DPI)
        self.fig.subplots_adjust(wspace=0.1, hspace=0.1)
        self.FUNCS_CFG = kwargs.get('funcs_cfg',FUNCS_CFG)
        dataset_list = kwargs.get('datasets',DB_DATASETS)
        self.DATASETS_MAP = dict(zip(dataset_list,[DATASETS_MAP[net] for net in dataset_list]))
        self.medianprops = dict(linestyle='-', linewidth=1, color='#522719')
        self.eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
        self.format = kwargs.get('format','png')
    
    def get_baseline(self, net, dataset):
        ent = DB_Utility.get_or_none(
            DB_Utility.func=='relu',
            DB_Utility.net==net,
            DB_Utility.dataset==dataset,
            DB_Utility.eps==None,
            )
        baseline = ent.test_acc
        return baseline

    def query_data(self, net, eps):
        funcs = [func['db_name'] for func in self.FUNCS_CFG]
        data = []
        for dataset in self.DATASETS_MAP.keys():
            baseline = DB_Utility.get_or_none(
                DB_Utility.net==net,
                DB_Utility.dataset==dataset,
                DB_Utility.eps==None,
                DB_Utility.func=='relu'   
            )
            ents = DB_Utility.select().where(
                DB_Utility.net==net,
                DB_Utility.dataset==dataset,
                DB_Utility.eps==eps,
                DB_Utility.func << funcs
            )
            data_per_dataset = [1-ent.test_acc/baseline.test_acc for ent in ents]
            data.extend(data_per_dataset)
        return data

    def set_axe_format(self,axe:Axes, title):
        axe.set_title(title)

        axe.xaxis.set_tick_params(labelsize=self.X_TICK_SIZE)
        axe.yaxis.set_tick_params(labelsize=self.Y_TICK_SIZE)
        axe.yaxis.set_label_position("right")
        axe.xaxis.set_label_position("top")
        axe.tick_params(axis='x', which='both', length=1, width=2)
        axe.set_ylim(-0.1,1.02)
        axe.yaxis.set_major_locator(ticker.FixedLocator([0,0.2,0.4,0.6,0.8,1]))
        # l,r = axe.get_xlim()
        # axe.set_xticks(ticks=[x+1 for x in range(len(self.eps))])
        axe.set_xticklabels(labels=self.eps)

    def plot_and_save(self):
        for x, (net, display_net) in enumerate(NETS_MAP.items()) :
            axe:Axes = self.axes_tuple[x]
            self.set_axe_format(axe,display_net)
            
            data = []
            for eps in self.eps:
                y_points = self.query_data(net,eps)
                data.append(y_points)
                print(net,eps,len(y_points))
            bp = axe.boxplot(data, patch_artist=True, medianprops=self.medianprops)

            for patch in bp['boxes']:
                patch.set_facecolor('lightblue') 

        self.fig.supxlabel('Privacy Budget', y=-0.01, fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel('Accuracy Loss', x=0.08, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(f'figure/netbox_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}.{self.format}',bbox_inches='tight')

class DataBoxPloter(NetBoxPloter):
    def __init__(self, dpi=300, **kwargs) -> None:
        super().__init__(dpi=dpi, **kwargs)
        
    def set_axe_format(self, axe: Axes, title):
        return super().set_axe_format(axe, title)

    def query_data(self, dataset, eps):
        funcs = [func['db_name'] for func in self.FUNCS_CFG]
        data = []
        for net in NETS_MAP.keys():
            baseline = DB_Utility.get_or_none(
                DB_Utility.net==net,
                DB_Utility.dataset==dataset,
                DB_Utility.eps==None,
                DB_Utility.func=='relu'   
            )
            ents = DB_Utility.select().where(
                DB_Utility.net==net,
                DB_Utility.dataset==dataset,
                DB_Utility.eps==eps,
                DB_Utility.func << funcs
            )
            data_per_net = [1-ent.test_acc/baseline.test_acc for ent in ents]
            data.extend(data_per_net)
        return data
    
    def plot_and_save(self):
        for x, (dataset, display_dataset) in enumerate(DATASETS_MAP.items()) :
            axe:Axes = self.axes_tuple[x]
            self.set_axe_format(axe,display_dataset)
            
            data = []
            for eps in self.eps:
                y_points = self.query_data(dataset,eps)
                data.append(y_points)
                print(dataset,eps,len(y_points))
            bp = axe.boxplot(data, patch_artist=True, medianprops=self.medianprops)

            for patch in bp['boxes']:
                patch.set_facecolor('lightblue') 

        self.fig.supxlabel('Privacy Budget', y=-0.01, fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel('Accuracy Loss', x=0.08, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(f'figure/databox_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}.{self.format}',bbox_inches='tight') 

class MIANetBoxPloter(NetBoxPloter):
    def __init__(
        self, 
        metric_type,
        attack_type,
        dpi=300,
        **kwargs
    ) -> None:
        super().__init__(dpi=dpi, **kwargs)
        self.metric_type = METRIC_DICT[metric_type] 
        self.attack_type = attack_type


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
            raise NotImplementedError('Invalid metric type')

    def set_axe_format(self,axe:Axes, title):
        axe.set_title(title)

        axe.xaxis.set_tick_params(labelsize=self.X_TICK_SIZE)
        axe.yaxis.set_tick_params(labelsize=self.Y_TICK_SIZE)
        axe.yaxis.set_label_position("right")
        axe.xaxis.set_label_position("top")
        axe.tick_params(axis='x', which='both', length=1, width=2)
        axe.set_ylim(-2.5,101)
        axe.yaxis.set_major_locator(ticker.FixedLocator([0,20,40,60,80,100]))
        # l,r = axe.get_xlim()
        # axe.set_xticks(ticks=[x+1 for x in range(len(self.eps))])
        axe.set_xticklabels(labels=self.eps)

    def query_data(self, net, eps, type):
        funcs = [func['db_name'] for func in self.FUNCS_CFG]
        data = []
        for dataset in self.DATASETS_MAP.keys():
            baseline = DB_Privacy.get_or_none(
                DB_Privacy.func=='relu',
                DB_Privacy.net==net, 
                DB_Privacy.dataset==dataset, 
                DB_Privacy.eps==None, 
                DB_Privacy.type==type, 
                DB_Privacy.shadow_dp==False,
            )
            assert baseline!=None, f'baseline of {net},{dataset} is not found.'
            print(f'baseline of {net},{dataset} is',self.get_metric(baseline))
            ents = DB_Privacy.select().where(
                DB_Privacy.net==net,
                DB_Privacy.dataset==dataset,
                DB_Privacy.eps==eps,
                DB_Privacy.func << funcs,
                DB_Privacy.type==type,
                DB_Privacy.shadow_dp==False,
            )
            data_per_dataset = [100*self.get_metric(ent)/(self.get_metric(baseline)+1e-8) for ent in ents]
            data.extend(data_per_dataset)
        return data

    def plot_and_save(self):
        for x, (net, display_net) in enumerate(NETS_MAP.items()) :
            axe:Axes = self.axes_tuple[x]
            self.set_axe_format(axe,display_net)
            
            data = []
            for eps in self.eps:
                y_points = self.query_data(net,eps,self.attack_type)
                data.append(y_points)
                print(f'Amount of {net},{eps} is',len(y_points))
            bp = axe.boxplot(data, patch_artist=True, medianprops=self.medianprops)

            for patch in bp['boxes']:
                patch.set_facecolor('lightblue') 

        self.fig.supxlabel('Privacy Budget', y=-0.01, fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel('Privacy Leakage(%)', x=0.08, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(f'figure/mia_netbox_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}.{self.format}',bbox_inches='tight')

class MIADataBoxPloter(MIANetBoxPloter):
    def __init__(self, metric_type, attack_type, dpi=300, **kwargs) -> None:
        super().__init__(metric_type, attack_type, dpi, **kwargs)
        net_list = kwargs.get('nets',DB_NETS)
        self.NETS_MAP = dict(zip(net_list,[NETS_MAP[net] for net in net_list]))
        self.fig , self.axes_tuple = plt.subplots(1,3, figsize=(self.SUB_PLOT_WIDTH,self.SUB_PLOT_HEIGHT),sharey='row',dpi=self.DPI)

    
    def query_data(self, dataset, eps, type):
        funcs = [func['db_name'] for func in self.FUNCS_CFG]
        data = []
        for net in self.NETS_MAP.keys():
            baseline = DB_Privacy.get_or_none(
                DB_Privacy.func=='relu',
                DB_Privacy.net==net, 
                DB_Privacy.dataset==dataset, 
                DB_Privacy.eps==None, 
                DB_Privacy.type==type, 
                DB_Privacy.shadow_dp==False,
            )
            assert baseline!=None, f'baseline of {net},{dataset} is not found.'
            print(f'baseline of {net},{dataset} is',self.get_metric(baseline))
            ents = DB_Privacy.select().where(
                DB_Privacy.net==net,
                DB_Privacy.dataset==dataset,
                DB_Privacy.eps==eps,
                DB_Privacy.func << funcs,
                DB_Privacy.type==type,
                DB_Privacy.shadow_dp==False,
            )
            data_per_dataset = [100*self.get_metric(ent)/(self.get_metric(baseline)+1e-8) for ent in ents]
            data.extend(data_per_dataset)
        return data

    def plot_and_save(self):
        for x, (dataset, display_dataset) in enumerate(self.DATASETS_MAP.items()) :
            axe:Axes = self.axes_tuple[x]
            self.set_axe_format(axe,display_dataset)
            
            data = []
            for eps in self.eps:
                y_points = self.query_data(dataset,eps,self.attack_type)
                data.append(y_points)
                print(f'Amount of {dataset},{eps} is',len(y_points))
            bp = axe.boxplot(data, patch_artist=True, medianprops=self.medianprops)

            for patch in bp['boxes']:
                patch.set_facecolor('lightblue') 

        self.fig.supxlabel('Privacy Budget', y=-0.01, fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel('Privacy Leakage(%)', x=0.08, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(f'figure/mia_databox_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}.{self.format}',bbox_inches='tight')

def main(args):
    # baseline 
    if(args.netbox):
        NetBoxPloter(funcs_cfg=FUNCS_DICT[args.funcs],format=args.format).plot_and_save()
    if(args.databox):
        DataBoxPloter(funcs_cfg=FUNCS_DICT[args.funcs],format=args.format).plot_and_save()
    if(args.miadatabox):
        MIADataBoxPloter(
            funcs_cfg=FUNCS_DICT[args.funcs],
            metric_type=args.metric,
            attack_type=args.type,
            format=args.format,
            datasets=args.datasets,
            nets=args.nets
        ).plot_and_save()
    if(args.mianetbox):
        MIANetBoxPloter(
            funcs_cfg=FUNCS_DICT[args.funcs],
            metric_type=args.metric,
            attack_type=args.type,
            format=args.format,
            datasets=args.datasets,
            nets=args.nets
        ).plot_and_save()
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--netbox','-nb', action='store_true')
    parser.add_argument('--databox','-db', action='store_true')
    parser.add_argument('--miadatabox','-mdb', action='store_true')
    parser.add_argument('--mianetbox','-mnb', action='store_true')

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