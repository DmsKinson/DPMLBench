import sys
import itertools

pwd = sys.path[0]

datasets=['mnist','fmnist','svhn','cifar10']
# datasets = ['mnist']
# nets = ['resnet','inception','vgg']
nets = ['simplenn']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000,None]

batchsize = 256
# n_epoch = {'mnist':60,'cifar10':90,'fmnist':60,'svhn':90}
n_epoch = 500

def gen_scripts(params):
    for param in params:
        net, dataset, e = param
        if(e is None):
            cmd = f'python3 {pwd}/main.py '+f" --net %s --dataset %s --batchsize 256 --epoch {n_epoch}" % param[:-1]
        else:
            cmd = f'python3 {pwd}/main.py '+f" --net %s --dataset %s --eps %s --batchsize 256 --epoch {n_epoch}" % param
        with open(sys.path[0]+'/../scripts/queue/'+"knn_%s_%s_%s.sh"%param,'wt') as f:
            f.write(cmd)

params = itertools.product(nets,datasets,eps)
# params = [
#     ('vgg','mnist',100), 
# ]
gen_scripts(params)