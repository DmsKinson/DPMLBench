import itertools
import sys
dir = sys.path[0]

# types = ['black','white','label']
types = ['label']
datasets=['mnist','fmnist','svhn','cifar10']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
# nets = ['simplenn','resnet','inception','vgg']
nets = ['simplenn']
funcs = [
        # 'relu',
        # 'tanh',
        # 'loss',
        # 'lp-2st',
        # 'adpclip',
        'knn',
        # 'alibi',
        # 'adp_alloc',
        # 'handcraft',
        'pate',
        # 'gep',
        # 'rgp',
        # 'dpgen'
    ]
shadow_dps = [False]

def gen_scripts(params):
    for param in params:
        type, func, net, dataset, e, shadow_dp = param
        # baseline: non-dp target and shadow 
        if(e is None):
            cmd = f'python3 -u {dir}/attack.py '+f"--type %s --func %s --net %s --dataset %s" % param[:-2];
        elif(shadow_dp):
            cmd = f'python3 -u {dir}/attack.py '+f"--type %s --func %s --net %s --dataset %s --eps %s --shadow_dp" % param[:-1]
        # non-dp shadow, dp target
        else: 
            cmd = f'python3 {dir}/attack.py '+f"--type %s --func %s --net %s --dataset %s --eps %s" % param[:-1]
        with open(sys.path[0]+'/../scripts/queue/'+"attack_%s_%s_%s_%s_%s_%s.sh"%param,'wt') as f:
            f.write(cmd)

params = itertools.product(types,funcs,nets,datasets,eps,shadow_dps)

# replace when need customization
params = [
('black', 'dpgen', 'simplenn', 'cifar10', 100, False),
('white', 'dpgen', 'simplenn', 'cifar10', 100, False),
('label', 'dpgen', 'simplenn', 'cifar10', 100, False),
('black', 'dpgen', 'resnet', 'cifar10', 100, False),
('white', 'dpgen', 'resnet', 'cifar10', 100, False),
('label', 'dpgen', 'resnet', 'cifar10', 100, False),
('black', 'dpgen', 'inception', 'cifar10', 100, False),
('white', 'dpgen', 'inception', 'cifar10', 100, False),
('label', 'dpgen', 'inception', 'cifar10', 100, False),
('black', 'dpgen', 'vgg', 'cifar10', 100, False),
('white', 'dpgen', 'vgg', 'cifar10', 100, False),
('label', 'dpgen', 'vgg', 'cifar10', 100, False),
('label', 'handcraft', 'inception', 'mnist', 1000, False),
('label', 'handcraft', 'vgg', 'svhn', 2, False),
]

gen_scripts(params)                 