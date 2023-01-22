import sys
import itertools
dir = sys.path[0]

nets = ['simplenn','resnet','inception','vgg']
datasets=['mnist','fmnist','svhn','cifar10']
# nets = ['simplenn']
# eps = [None,0.2,0.3,0.4,0.5,1,2,4,8,100,1000]
eps = [None]
batchsize = 256
n_epoch = {'mnist':60,'cifar10':90,'fmnist':60,'svhn':90}
def gen_scripts(params):
    for param in params:
        net, dataset, e = param
        if(e is None):
            cmd = f'python3 {dir}/train_shadows.py '+f" --net %s --dataset %s --batchsize 256 --epoch {n_epoch[dataset]}" % param[:-1]
        else:
            cmd = f'python3 {dir}/train_shadows.py '+f" --net %s --dataset %s -p --eps %s --batchsize 256 --epoch {n_epoch[dataset]}" % param
        with open(sys.path[0]+'/../scripts/queue/'+"shadow_%s_%s_%s.sh"%param,'wt') as f:
            f.write(cmd)
        
params = itertools.product(nets, datasets, eps)
# params = [

# ]
gen_scripts(params)
