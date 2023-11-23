import sys,os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

datasets=['mnist','fmnist','svhn','cifar10']
datasets=['cifar10']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
eps = [100]

def gen_scripts(params):
    for param in params:
        dataset, e,  = param
        if(e is None):
            cmd = f'python3 -u {pwd}/main.py '+f"--dataset %s --ni --sample" % param[:-1]
        else:
            cmd = f'python3 -u {pwd}/main.py '+f"--dataset %s --eps %s --ni --sample" % param
        with open(pwd+'/../scripts/queue/'+"dp_gen_sample_%s_%s.sh"%param,'wt') as f:
            f.write(cmd)

params = itertools.product( datasets, eps)
params = [
    # ('mnist',1),
    ('cifar10',4)
]
gen_scripts(params)