import sys,os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

datasets=['MNIST','FashionMNIST','SVHN','CIFAR10']
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000]

def gen_scripts(params):
    for param in params:
        dataset, e,  = param
        cmd = f"docker run --rm --gpus all -v $(pwd):/workspace -e CUDA_VISIBLE_DEVICES={{}} -v /data2/zmh/dataset:/data2/zmh/dataset -u $(id -u):$(id -g)  pytorch/pytorch:1.7.1-privateset python3 private-set/main.py  --enable_privacy --target_epsilon {e} --model ConvNet --dataset {dataset} --spc 10"
        with open(os.path.join(pwd, '..', '..', 'scripts', "pri-set_train_%s_%s.sh"%param),'wt') as f:
            f.write(cmd)

params = itertools.product( datasets, eps)
# params = [ 
#     ('cifar10',4)
# ]
gen_scripts(params)