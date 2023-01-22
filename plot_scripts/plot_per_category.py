from name_map import NET_MAP, DATASET_MAP 
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes

DATA_PREPARATION = [
    'dpgen','gswgan','handcraft','relu'
]

MODEL_DESIGN = [
    'loss','tanh','relu'
]

MODEL_TRAINING = [
    'relu','tanh','adpclip','gep','rgp','adp_alloc','relu'
]

MODEL_ENSEMBLE = [
    'pate','knn','relu'
]

class AccPloter():
    def __init__(self,dpi=300, **kwargs) -> None:
        self.X_TICK_SIZE = 10
        self.Y_TICK_SIZE = 8
        self.SUB_PLOT_WIDTH = 4
        self.SUB_PLOT_HEIGHT = 2.5
        self.SUPLABEL_SIZE = 16
        self.LABEL_SIZE = 12
        self.DPI = dpi
        self.inf_eps = 10000

        self.FUNCS_CFG = kwargs.get('funcs_cfg',FUNCS_CFG)
        self.NETS_MAP = kwargs.get('nets_map',NET_MAP)
        self.DATASETS_MAP = kwargs.get('datasets_map',DATASET_MAP)
        self.n_col = len(self.NETS_MAP)
        self.n_row = len(self.DATASETS_MAP)
        self.fig_width = self.SUB_PLOT_WIDTH * self.n_col;
        self.fig_height = self.SUB_PLOT_HEIGHT * self.n_row;
        self.eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
        self.xlabels = self.eps
        plt.style.use('ggplot')
        self.fig , self.axes_tuple = plt.subplots(self.n_row,self.n_col, figsize=(self.fig_width,self.fig_height),sharex='col',sharey='row',dpi=self.DPI)
        self.fig.subplots_adjust(wspace=0.1, hspace=0.1)
        self.format = kwargs.get('format','png')

    def set_axe_format(self, x, net, y, dataset, axe:Axes):
        axe.set_xscale('log')
        axe.xaxis.set_tick_params(labelsize=self.X_TICK_SIZE)
        axe.yaxis.set_tick_params(labelsize=self.Y_TICK_SIZE)
        axe.yaxis.set_label_position("right")
        axe.xaxis.set_label_position("top")
        axe.set_ylim(-1,101)
        axe.set_xlim(0.15,1002)
        axe.set_xticks(ticks=self.eps)
        axe.set_xticklabels(self.xlabels, rotation=300, ha="center",va='center_baseline')
        axe.xaxis.set_major_formatter(ticker.ScalarFormatter())
        axe.xaxis.set_minor_locator(ticker.NullLocator())
        axe.xaxis.set_major_locator(ticker.FixedLocator(self.eps))
        axe.xaxis.set_major_formatter(ticker.FixedFormatter(self.xlabels))
        axe.yaxis.set_major_locator(ticker.FixedLocator([0,20,40,60,80,100]))
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
                    length = len(ypoints)
                    axe.plot(self.eps[:length],ypoints,marker=config['marker'], markersize=4, c=config['color'], label=config.get('display_name',config['db_name']))
        # get legends from simple-mnist setting, which has all funcs
        handles, labels = self.axes_tuple[0,0].get_legend_handles_labels()
        self.fig.legend(handles, labels, loc='center',bbox_to_anchor=(0.5,0.03),labelspacing=1,ncol=len(self.FUNCS_CFG))
        self.fig.supxlabel('Privacy Budget', y=0.06, fontsize=self.SUPLABEL_SIZE)
        self.fig.supylabel('Accuracy', x=0.08, fontsize=self.SUPLABEL_SIZE)
        self.fig.savefig(f'figure/acc_{time.strftime("%m_%d_%H:%M:%S",time.localtime())}.png',bbox_inches='tight')

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
                DB_Utility.type=='target'
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