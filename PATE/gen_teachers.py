import sys
pwd = sys.path[0]
nets = ['simplenn']
datasets = ['fmnist','svhn','cifar10','mnist']
# epoch = {'mnist':60,'cifar10':90,'fmnist':60,'svhn':90}
epoch = 500

n_teacher = 100
for net in nets:
    for dataset in datasets:
        for teacher_id in range(n_teacher):
            params = (net, dataset, epoch, n_teacher, teacher_id)
            cmd = f'python3 {pwd}/train_single_teacher.py --net %s --dataset %s --epoch %s --batchsize 256 --n_teacher %s --teacher_id %s' % params
            with open(pwd+'/../scripts/queue/'+f"pate_{net}_{dataset}_{teacher_id}th.sh",'wt') as f:
                f.write(cmd)