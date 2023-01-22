import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

from db_models import DB_Privacy
from itertools import product

TEMPLATE = f"""
\\begin{{tabular}}{{ccccc}}
    \\toprule
          &  FMNIST & SVHN  & CIFAR-10 \\\\
    \\midrule
    \\textbf{{SimpleNet}}(0.17M) & %s & %s & %s \\\\
    \\textbf{{ResNet}}(0.26M) & %s & %s & %s \\\\
    \\textbf{{InceptionNet}}(1.97M) & %s & %s & %s \\\\
    \\textbf{{VGG}}(128.8M) & %s & %s & %s \\\\
    \\bottomrule
\\end{{tabular}}
"""
values = []

nets = ["simplenn","resnet","inception","vgg"]
datasets = ["fmnist","svhn","cifar10"]
types = ["black","white","label"]

params = product(nets,datasets,types)
line = ""
cnt = 0
for param in params:
    net,dataset,type = param
    ent = DB_Privacy.get_or_none(
        DB_Privacy.func=="relu",
        DB_Privacy.net==net,
        DB_Privacy.dataset==dataset,
        DB_Privacy.eps==None,
        DB_Privacy.type==type,
        DB_Privacy.extra==None,
        
    )
    assert ent is not None, f"{net},{dataset},{type} is none"
    metric = max(0.5,ent.auc)
    line += f"{metric:.2f}/"
    cnt+=1
    if(cnt==3):
        values.append(line[:-1])
        line = ""
        cnt = 0

assert len(values)==len(nets)*len(datasets), f"Actual len(values)={len(values)}"
values = tuple(values)
latex_out = TEMPLATE % values

with open(os.path.join(pwd,"baseline_mia.tex"),"wt") as f:
    f.write(latex_out)

