import os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

datasets=['mnist','fmnist','cifar10','svhn']
nets = ['simple','resnet','vgg','inception']
nets = ['simplenn']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]

# eps = [0.2,0.3,0.4,0.5,1,2,4,8]
# eps = [100, 1000]

batchsize = 256
n_epoch = {'mnist':200,'cifar10':200,'fmnist':200,'svhn':200}

def gen_scripts(params):
    for param in params:
        net, dataset, e = param
        cmd = f"python3 {os.path.join(pwd, 'main.py')} "+f" --net %s --dataset %s --eps %s --epoch {n_epoch[dataset]}" % param
        with open(os.path.join(pwd,'..','..','scripts',"lp-mst_%s_%s_%s.sh"%param),'wt') as f:
            f.write(cmd)

params = itertools.product(nets, datasets, eps)
# params = [

# ]
gen_scripts(params)
