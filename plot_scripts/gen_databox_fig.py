import sys,os
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd,'..')) 

import argparse
from matplotlib import ticker
from matplotlib.axes import Axes
from db_models import  DB_Privacy, DB_Utility
import matplotlib.pyplot as plt
import time
from name_map import NETS_MAP, DATASETS_MAP
from ploter_static import FUNCS_CFG,DB_DATASETS,DB_NETS,METRIC_DICT,MetricType,FUNCS_DICT
from peewee import *

DATABASE_DIR = os.path.join(pwd,'..','database')
if(not os.path.exists(DATABASE_DIR)):
    os.mkdir(DATABASE_DIR)

ROE1_PATH = os.path.join(DATABASE_DIR, 'main_1roe.db')
ROE2_PATH = os.path.join(DATABASE_DIR, 'main_2roe.db')
ROE3_PATH = os.path.join(DATABASE_DIR, 'main.db')
db1 = SqliteDatabase(ROE1_PATH)
db2 = SqliteDatabase(ROE2_PATH)
db3 = SqliteDatabase(ROE3_PATH)

class DB_Uitlity_1roe(DB_Utility):
    class Meta:
        database = db1
        table_name = 'Utility'

class DB_Utility_2roe(DB_Utility):
    class Meta:
        database = db2
        table_name = 'Utility'

class DB_Uitlity_3roe(DB_Utility):
    class Meta:
        database = db3
        table_name = 'Utility'

class DB_Privacy_1roe(DB_Privacy):
    class Meta:
        database = db1
        table_name = 'Privacy'

class DB_Privacy_2roe(DB_Privacy):
    class Meta:
        database = db2
        table_name = 'Privacy'

class DB_Privacy_3roe(DB_Privacy):
    class Meta:
        database = db3
        table_name = 'Privacy'

class NetBoxPloter():
    def __init__(self, dpi=300,**kwargs) -> None:
        plt.style.use('ggplot')
        self.X_TICK_SIZE = 10
        self.Y_TICK_SIZE = 8
        self.SUB_PLOT_WIDTH = 4.5
        self.SUB_PLOT_HEIGHT = 2
        self.SUPLABEL_SIZE = 12
        self.LABEL_SIZE = 12
        self.DPI = dpi
        self.YLABEL_XPOS = 0.1
        self.FUNCS_CFG = kwargs.get('funcs_cfg',FUNCS_CFG)
        dataset_list = kwargs.get('datasets',DB_DATASETS)
        self.n_col = 4
        self.BOX_WIDTH = 0.4
        self.fig , self.axes_tuple = plt.subplots(1,4, figsize=(self.n_col*self.SUB_PLOT_WIDTH,self.SUB_PLOT_HEIGHT),sharey='row',dpi=self.DPI)
        self.fig.subplots_adjust(wspace=0.03, hspace=0.1)
        self.DATASETS_MAP = dict(zip(dataset_list,[DATASETS_MAP[net] for net in dataset_list]))
        self.medianprops = dict(linestyle='-', linewidth=1, color='#522719')
        self.flierprops = dict(marker='o',markersize=4)
        self.eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
        self.format = kwargs.get('format','png')
        self.db_models = [DB_Uitlity_1roe,DB_Utility_2roe,DB_Uitlity_3roe]
        self.name = self.name = kwargs['name'] if kwargs['name'] is not None else f'netbox_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}'
        self.filename = os.path.join(pwd,'..','figure',f'{self.name}.{self.format}')
    
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
        funcs.remove('pate')
        funcs.remove('knn')
        funcs_pates = ['knn','pate']
        data = []
        for dataset in self.DATASETS_MAP.keys():
            for DB in self.db_models:
                baseline = DB.get_or_none(
                    DB.net==net,
                    DB.dataset==dataset,
                    DB.eps==None,
                    DB.func=='relu'   
                )
                ents = DB.select().where(
                    DB.net==net,
                    DB.dataset==dataset,
                    DB.eps==eps,
                    DB.func << funcs,
                    DB.extra==None,
                )
                data_per_dataset = [(1-ent.test_acc/baseline.test_acc)*100 for ent in ents]
                data.extend(data_per_dataset)
                ents = DB.select().where(
                    DB.net==net,
                    DB.dataset==dataset,
                    DB.eps==eps,
                    DB.func << funcs_pates,
                    DB.extra=='uda',
                )
                data_per_dataset = [(1-ent.test_acc/baseline.test_acc)*100 for ent in ents]
                data.extend(data_per_dataset)

        return data

    def set_axe_format(self,axe:Axes, title):
        axe.set_title(title, fontdict={'fontsize':self.LABEL_SIZE})

        axe.xaxis.set_tick_params(labelsize=self.X_TICK_SIZE)
        axe.yaxis.set_tick_params(labelsize=self.Y_TICK_SIZE)
        axe.yaxis.set_label_position("right")
        axe.xaxis.set_label_position("top")
        axe.tick_params(axis='x', which='both', length=1, width=2)
        axe.tick_params(axis='both', which='both', direction='in')
        axe.set_ylim(-1,102)
        axe.yaxis.set_major_locator(ticker.FixedLocator([0,20,40,60,80,100]))
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
            bp = axe.boxplot(data, patch_artist=True, medianprops=self.medianprops, flierprops=self.flierprops, widths=self.BOX_WIDTH)

            for patch in bp['boxes']:
                patch.set_facecolor('lightblue') 

        self.fig.supylabel('Utility Loss(%)', x=self.YLABEL_XPOS, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(self.filename,bbox_inches='tight')

class DataBoxPloter(NetBoxPloter):
    def __init__(self, dpi=300, **kwargs) -> None:
        super().__init__(dpi=dpi, **kwargs)
        self.name = kwargs['name'] if kwargs['name'] is not None else f'databox_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}'
        self.filename = os.path.join(pwd,'..','figure',f'{self.name}.{self.format}')

    def query_data(self, dataset, eps):
        funcs = [func['db_name'] for func in self.FUNCS_CFG]
        funcs.remove('pate')
        funcs.remove('knn')
        funcs_pate = ['knn','pate']
        data = []
        for net in NETS_MAP.keys():
            for DB in self.db_models:
                baseline = DB.get_or_none(
                    DB.net==net,
                    DB.dataset==dataset,
                    DB.eps==None,
                    DB.func=='relu'   
                )
                ents = DB.select().where(
                    DB.net==net,
                    DB.dataset==dataset,
                    DB.eps==eps,
                    DB.func << funcs,
                    DB.extra==None
                )
                data_per_net = [(1-ent.test_acc/baseline.test_acc)*100 for ent in ents]
                data.extend(data_per_net)
                ents = DB.select().where(
                    DB.net==net,
                    DB.dataset==dataset,
                    DB.eps==eps,
                    DB.func << funcs_pate,
                    DB.extra=='uda'
                )
                data_per_net = [(1-ent.test_acc/baseline.test_acc)*100 for ent in ents]
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
            bp = axe.boxplot(data, patch_artist=True, medianprops=self.medianprops, flierprops=self.flierprops, widths=self.BOX_WIDTH)

            for patch in bp['boxes']:
                patch.set_facecolor('lightblue') 

        # self.fig.supxlabel('Privacy Budget', y=-0.01, fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel('Utility Loss(%)', x=self.YLABEL_XPOS, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(self.filename,bbox_inches='tight') 

class MIANetBoxPloter(NetBoxPloter):
    def __init__(
        self, 
        metric_type,
        attack_types,
        dpi=300,
        **kwargs
    ) -> None:
        super().__init__(dpi=dpi, **kwargs)
        self.name = kwargs['name'] if kwargs['name'] is not None else f'mia_netbox_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}'
        self.filename = os.path.join(pwd,'..','figure',f'{self.name}.{self.format}')
        self.metric_type = METRIC_DICT[metric_type] 
        self.attack_types = attack_types
        self.db_models = [DB_Privacy_1roe,DB_Privacy_2roe,DB_Privacy_3roe]


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

    def set_axe_format(self,axe:Axes, title):
        axe.set_title(title, fontdict={'fontsize':self.LABEL_SIZE})

        axe.xaxis.set_tick_params(labelsize=self.X_TICK_SIZE)
        axe.yaxis.set_tick_params(labelsize=self.Y_TICK_SIZE)
        axe.yaxis.set_label_position("right")
        axe.xaxis.set_label_position("top")
        axe.tick_params(axis='x', which='both', length=1, width=2)
        axe.tick_params(axis='both', which='both', direction='in')
        axe.set_ylim(-1,101)
        axe.yaxis.set_major_locator(ticker.FixedLocator([0,20,40,60,80,100]))
        # l,r = axe.get_xlim()
        # axe.set_xticks(ticks=[x+1 for x in range(len(self.eps))])
        axe.set_xticklabels(labels=self.eps)

    def query_data(self, net, eps, type):
        funcs = [func['db_name'] for func in self.FUNCS_CFG]
        try:
            funcs.remove('knn')
            funcs.remove('pate')
        except:
            pass
        funcs_pate = ['knn','pate']
        data = []
        for dataset in self.DATASETS_MAP.keys():
            for DB in self.db_models:
                baseline = DB.get_or_none(
                    DB.func=='relu',
                    DB.net==net, 
                    DB.dataset==dataset, 
                    DB.eps==None, 
                    DB.type==type, 
                    DB.shadow_dp==False,
                    DB.extra==None
                )
                assert baseline!=None, f'baseline of {net},{dataset},{type} is not found.'
                print(f'baseline of {net},{dataset},{type} is',self.get_metric(baseline))
                ents = DB.select().where(
                    DB.net==net,
                    DB.dataset==dataset,
                    DB.eps==eps,
                    DB.func << funcs,
                    DB.type==type,
                    DB.shadow_dp==False,
                    DB.extra==None
                )
                data_per_dataset = [100*self.get_metric(ent)/(self.get_metric(baseline)+1e-8) for ent in ents]
                data.extend(data_per_dataset)
                ents = DB.select().where(
                    DB.net==net,
                    DB.dataset==dataset,
                    DB.eps==eps,
                    DB.func << funcs_pate,
                    DB.type==type,
                    DB.shadow_dp==False,
                    DB.extra=='uda'
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
                y_points = []
                for attack_type in self.attack_types:
                    points = self.query_data(net,eps,attack_type)
                    y_points.extend(points)
                data.append(y_points)
                print(f'Amount of {net},{eps} is',len(y_points))
            bp = axe.boxplot(data, patch_artist=True, medianprops=self.medianprops, flierprops=self.flierprops, widths=self.BOX_WIDTH)

            for patch in bp['boxes']:
                patch.set_facecolor('lightblue') 

        # self.fig.supxlabel('Privacy Budget', y=-0.01, fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel('Privacy Leakage(%)', x=self.YLABEL_XPOS, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(self.filename,bbox_inches='tight')

class MIADataBoxPloter(MIANetBoxPloter):
    def __init__(self, metric_type, attack_types, dpi=300, **kwargs) -> None:
        super().__init__(metric_type, attack_types, dpi, **kwargs)
        self.name = kwargs['name'] if kwargs['name'] is not None else f'mia_databox_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}'
        self.filename = os.path.join(pwd,'..','figure',f'{self.name}.{self.format}')
        net_list = kwargs.get('nets',DB_NETS)
        self.NETS_MAP = dict(zip(net_list,[NETS_MAP[net] for net in net_list]))
        self.n_col = len(net_list)
        self.fig , self.axes_tuple = plt.subplots(1,self.n_col, figsize=(self.n_col*self.SUB_PLOT_WIDTH,self.SUB_PLOT_HEIGHT),sharey='row',dpi=self.DPI)
        self.fig.subplots_adjust(wspace=0.03,hspace=0.01)
        self.YLABEL_XPOS = 0.1
    
    def query_data(self, dataset, eps, type):
        funcs = [func['db_name'] for func in self.FUNCS_CFG]
        funcs.remove('knn')
        funcs.remove('pate')
        funcs_pate = ['knn','pate']
        data = []
        for net in self.NETS_MAP.keys():
            for DB in self.db_models:
                baseline = DB.get_or_none(
                    DB.func=='relu',
                    DB.net==net, 
                    DB.dataset==dataset, 
                    DB.eps==None, 
                    DB.type==type, 
                    DB.shadow_dp==False,
                    DB.extra == None
                )
                assert baseline!=None, f'baseline of {net},{dataset},{type} is not found.'
                print(f'baseline of {net},{dataset},{type} is',self.get_metric(baseline))
                ents = DB.select().where(
                    DB.net==net,
                    DB.dataset==dataset,
                    DB.eps==eps,
                    DB.func << funcs,
                    DB.type==type,
                    DB.shadow_dp==False,
                    DB.extra==None
                )
                data_per_dataset = [100*self.get_metric(ent)/(self.get_metric(baseline)+1e-8) for ent in ents]
                data.extend(data_per_dataset)
                ents = DB.select().where(
                    DB.net==net,
                    DB.dataset==dataset,
                    DB.eps==eps,
                    DB.func << funcs_pate,
                    DB.type==type,
                    DB.shadow_dp==False,
                    DB.extra=='uda'
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
                y_points = []
                for attack_type in self.attack_types:
                    points = self.query_data(dataset,eps,attack_type)
                    y_points.extend(points)
                data.append(y_points)
                print(f'Amount of {dataset},{eps} is',len(y_points))
            bp = axe.boxplot(data, patch_artist=True, medianprops=self.medianprops, widths=self.BOX_WIDTH, flierprops=self.flierprops)

            for patch in bp['boxes']:
                patch.set_facecolor('lightblue') 

        # self.fig.supxlabel('Privacy Budget', y=-0.01, fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel('Privacy Leakage(%)', x=self.YLABEL_XPOS, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(self.filename,bbox_inches='tight')

def main(args):
    # baseline 
    if(args.netbox):
        NetBoxPloter(
            funcs_cfg=FUNCS_DICT[args.funcs],
            format=args.format,
            name=args.name,
        ).plot_and_save()
    if(args.databox):
        DataBoxPloter(
            funcs_cfg=FUNCS_DICT[args.funcs],
            format=args.format,
            name=args.name,  
        ).plot_and_save()
    if(args.miadatabox):
        MIADataBoxPloter(
            funcs_cfg=FUNCS_DICT[args.funcs],
            metric_type=args.metric,
            attack_types=args.types,
            format=args.format,
            datasets=args.datasets,
            nets=args.nets,
            name=args.name
        ).plot_and_save()
    if(args.mianetbox):
        MIANetBoxPloter(
            funcs_cfg=FUNCS_DICT[args.funcs],
            metric_type=args.metric,
            attack_types=args.types,
            format=args.format,
            datasets=args.datasets,
            nets=args.nets,
            name=args.name
        ).plot_and_save()
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--netbox','-nb', action='store_true')
    parser.add_argument('--databox','-db', action='store_true')
    parser.add_argument('--miadatabox','-mdb', action='store_true')
    parser.add_argument('--mianetbox','-mnb', action='store_true')

    parser.add_argument('--nets', type=str, nargs='+', help='nets list to plot', choices=['simplenn','resnet','inception','vgg'])
    parser.add_argument('--datasets', type=str, nargs='+', help='datasets list to plot',choices=['mnist','fmnist','svhn','cifar10'])
    parser.add_argument('--types', type=str, nargs='+',choices=['black','white','label'])
    parser.add_argument('--metric', type=str, choices=list(METRIC_DICT.keys()), default='auc_norm')
    parser.add_argument('--funcs', type=str, default='all', choices=list(FUNCS_DICT.keys()))
    parser.add_argument('--format', type=str, default='png', choices=['png','pdf','jpg'])
    parser.add_argument('--name',type=str, help='file name without extensive name')
    args = parser.parse_args()
    if(args.nets==None):
        args.nets = DB_NETS
    if(args.datasets == None):
        args.datasets = DB_DATASETS
    if(args.types==None):
        args.types = ['black','white']
    main(args)