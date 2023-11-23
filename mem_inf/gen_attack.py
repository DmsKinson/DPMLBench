import os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

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
            cmd = f"python3 -u {os.path.join(pwd, 'attack.py')} "+f"--type %s --func %s --net %s --dataset %s" % param[:-2];
        elif(shadow_dp):
            cmd = f"python3 -u {os.path.join(pwd, 'attack.py')} "+f"--type %s --func %s --net %s --dataset %s --eps %s --shadow_dp" % param[:-1]
        # non-dp shadow, dp target
        else: 
            cmd = f"python3 {os.path.join(pwd, 'attack.py')} "+f"--type %s --func %s --net %s --dataset %s --eps %s" % param[:-1]
        with open(os.path.join(pwd,'..','..','scripts',"attack_%s_%s_%s_%s_%s_%s.sh"%param),'wt') as f:
            f.write(cmd)

params = itertools.product(types,funcs,nets,datasets,eps,shadow_dps)

# replace when need customization
params = [

]

gen_scripts(params)                 