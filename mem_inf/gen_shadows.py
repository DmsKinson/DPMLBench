import os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

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
            cmd = f"python3 {os.path.join(pwd, 'train_shadows.py')} "+f" --net %s --dataset %s --batchsize 256 --epoch {n_epoch[dataset]}" % param[:-1]
        else:
            cmd = f"python3 {os.path.join(pwd, 'train_shadows.py')} "+f" --net %s --dataset %s -p --eps %s --batchsize 256 --epoch {n_epoch[dataset]}" % param
        with open(os.path.join(pwd,'..','..','scripts',"shadow_%s_%s_%s.sh"%param),'wt') as f:
            f.write(cmd)
        
params = itertools.product(nets, datasets, eps)
params = [
    # (net, dataset, eps)
    ('resnet', 'cifar10', 1)
]
gen_scripts(params)
