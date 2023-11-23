import os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

# datasets=['mnist','fmnist','svhn','cifar10']
datasets = ['svhn','cifar10']
nets = ['simplenn','resnet','inception','vgg']
# nets = ['simplenn']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]

batchsize = 256
n_epoch = {'mnist':50,'cifar10':90,'fmnist':60,'svhn':90}
n_query = {'mnist':700,'cifar10':4000,'svhn':3000,'fmnist':1000}
# n_epoch = 500

def gen_scripts(params):
    for param in params:
        net, dataset, e = param
        if(e is None):
            cmd = f'python3 {pwd}/uda_main.py '+f" --net %s --dataset %s --batchsize 256 --epoch {n_epoch[dataset]} --n_query {n_query[dataset]}" % param[:-1]
        else:
            cmd = f'python3 {pwd}/uda_main.py '+f" --net %s --dataset %s --eps %s --batchsize 256 --epoch {n_epoch[dataset]}  --n_query {n_query[dataset]}" % param
        with open(os.path.join(pwd,'..','..','scripts',"uda_knn_%s_%s_%s.sh"%param),'wt') as f:
            f.write(cmd)

params = itertools.product(nets,datasets,eps)
# params = [
#     ('vgg','mnist',100), 
# ]
gen_scripts(params)