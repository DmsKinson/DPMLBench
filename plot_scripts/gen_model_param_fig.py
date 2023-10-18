import os,sys
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd,'..')) 

from models import get_model
from db_models import DB_Csv
import matplotlib.pyplot as plt
import pandas as pd
from tools import get_model_parameters_amount

def get_test_acc(net,dataset,eps):
    ent = DB_Csv.get_or_none(
        DB_Csv.func=='relu',
        DB_Csv.dataset==dataset,
        DB_Csv.net == net,
        DB_Csv.eps == eps,
        DB_Csv.extra == None
    )
    if ent == None :
        return None
    assert ent != None, f'{net}_{eps} is not existed.'
    csv_df = pd.read_csv(ent.location)
    csv_df.columns = ['epoch','trainloss','trainacc','testloss','testacc','traincost','testcost']
    test_acc = csv_df['testacc']
    return test_acc.iat[-1]

plt.style.use('ggplot')
nets = ['resnet','resnet44','resnet74','resnet110','resnet1202']

eps = [0.2,0.5,1,8,100,1000]
datasets = ['fmnist','cifar10']
colors = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b'
]
markers = [
    'o','s','P','v','*','D'
]
font_size = 16
marker_size = 8
fig ,axes = plt.subplots(1,2,figsize=(10,4),sharey='row')
fig.subplots_adjust(wspace=0.05)
fig.supxlabel('Number of Parameters(M)',y=-0.03,fontsize=font_size)
fig.supylabel('Accuracy(%)',x=0.07,fontsize=font_size)
fig.set_label('epsilon')
xs = range(len(nets))
xlabels = [get_model_parameters_amount(get_model(net))/1e6 for net in nets]
for idx,(dataset,axe) in enumerate(zip(datasets,axes)) :
    axe.set_xticks(xs,labels=[f'{x:.2f}' for x in xlabels])    
    axe.set_title(dataset.upper())
    for e,color,marker in zip(eps,colors,markers):
        ys = [get_test_acc(net,dataset,e) for net in nets]
        print(dataset,e,ys)
        axe.plot(xs[:len(ys)], ys,label=e, markerfacecolor='white',marker=marker, markersize=marker_size,c=color)
handles,labels = axe.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=font_size, loc='lower center',bbox_to_anchor=(0.5,-0.17),labelspacing=1,ncol=len(eps))
fig.savefig(os.path.join(pwd,'..','figure','acc-params.pdf'),bbox_inches='tight')
        