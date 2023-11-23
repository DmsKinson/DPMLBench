import sys,os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

datasets=['mnist','fmnist','svhn','cifar10']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]


def gen_scripts(params):
    for param in params:
        dataset, e,  = param
        if(e is None):
            cmd = f"python3 -u {os.path.join(pwd, 'main.py')} "+f"--dataset %s --ni" % param[:-1]
        else:
            cmd = f"python3 -u {os.path.join(pwd, 'main.py')} "+f"--dataset %s --eps %s --ni" % param
        with open(os.path.join(pwd,'..','..','scripts',"dp_gen_train_%s_%s.sh"%param),'wt') as f:
            f.write(cmd)

params = itertools.product( datasets, eps)
params = [ 
    ('cifar10',4)
]
gen_scripts(params)