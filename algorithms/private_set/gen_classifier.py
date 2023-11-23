import sys,os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

nets = ['simplenn','resnet','inception','vgg']
datasets = ['mnist','fmnist','svhn','cifar10']
seed = 53458

# datasets = ['cifar10']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
# eps = [100]

def gen_scripts(params):
    for param in params:
        net, dataset, e,  = param
        cmd = f"python3 -u {os.path.join(pwd, 'train_classifier.py')} " + f"--net %s --dataset %s --eps %s --seed {seed}" % param
        with open(os.path.join(pwd, '..', '..', 'scripts', "pri-set_clsfer_%s_%s_%s.sh"%param),'wt') as f:
            f.write(cmd)

params = itertools.product(nets, datasets, eps)
# params = [ 
#     # ('vgg','mnist',0.3),
#     ('vgg','mnist',0.4),
#     # ('vgg','mnist',1),
#     # ('vgg','mnist',4),
#     # ('vgg','mnist',8),
#     # ('vgg','mnist',1000)
# ]
gen_scripts(params)