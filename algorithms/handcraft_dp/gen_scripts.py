import os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

datasets=['mnist','fmnist','svhn','cifar10']
# nets = ['simple','resnet','inception','vgg']
nets = ['linear']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]

batchsize = 256
n_epoch = {'mnist':60,'cifar10':90,'fmnist':60,'svhn':90}

def gen_scripts(params):
    for param in params:
        net, dataset, e,  = param
        if(e is None):
            cmd = f"python3 -u {os.path.join(pwd, 'main.py')} "+f" --net %s --dataset %s --epoch {n_epoch[dataset]} --batchsize {batchsize}" % param[:-1]
        else:
            cmd = f"python3 -u {os.path.join(pwd, 'main.py')} "+f" -p --net %s --dataset %s  --eps %s --epoch {n_epoch[dataset]} --batchsize {batchsize}" % param
        with open(os.path.join(pwd,'..','..','scripts',"hand_%s_%s_%s.sh"%param),'wt') as f:
            f.write(cmd)

params = itertools.product(nets, datasets, eps)
# params = [
#     ('inception','svhn',0.5),
#     ('inception','svhn',2)
# ]
gen_scripts(params)