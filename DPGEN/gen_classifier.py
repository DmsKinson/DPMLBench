import sys,os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

nets = ['simplenn','resnet','inception','vgg']
datasets = ['mnist','fmnist','svhn','cifar10']
# datasets = ['cifar10']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
# eps = [100]

def gen_scripts(params):
    for param in params:
        net, dataset, e,  = param
        cmd = f'python3 -u {pwd}/train_classifier.py '+f"--net %s --dataset %s --eps %s " % param
        with open(pwd+'/../scripts/queue/'+"dp_gen_clsfer_%s_%s_%s.sh"%param,'wt') as f:
            f.write(cmd)

params = itertools.product(nets, datasets, eps)
# params = [ 
# ]
gen_scripts(params)