import os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

nets = ['simplenn']
datasets = ['fmnist','svhn','cifar10','mnist']
# epoch = {'mnist':60,'cifar10':90,'fmnist':60,'svhn':90}
epoch = 500

n_teacher = 100
def gen_scripts(params):
    for p in params:
        net, dataset, teacher_id = p
        params = (net, dataset, epoch, n_teacher, teacher_id)
        cmd = f'python3 {pwd}/train_single_teacher.py --net %s --dataset %s --epoch %s --batchsize 256 --n_teacher %s --teacher_id %s' % params
        with open(os.path.join(pwd,'..','..','scripts',f"pate_teacher_{net}_{dataset}_{teacher_id}th.sh"),'wt') as f:
            f.write(cmd)

params = itertools.product(nets, datasets, list(range(n_teacher)))
gen_scripts(params)