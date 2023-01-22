import sys
import itertools
pwd = sys.path[0]

datasets=['mnist','fmnist','svhn','cifar10']
# nets = ['simple','resnet','inception','vgg']
nets = ['simplenn']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
# eps = [100,1000]

batchsize = 256
n_epoch = {'mnist':60,'cifar10':90,'fmnist':60,'svhn':90}


def gen_scripts(params):
    for param in params:
        net, dataset, e,  = param
        cmd = f'python3 {pwd}/main.py '+f" --net %s --dataset %s --eps %s --batchsize 256 --epoch {n_epoch[dataset]}" % param
        with open(sys.path[0]+'/../scripts/queue/'+"adpAlloc_%s_%s_%s.sh"%param,'wt') as f:
            f.write(cmd)

params = itertools.product(nets,datasets,eps)
# params = [
#     ('simple','fmnist',0.4),
#     ('simple','fmnist',0.3),
#     ('simple','fmnist',0.2),
    
# ]
gen_scripts(params)