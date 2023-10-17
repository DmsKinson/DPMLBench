# DPMLBench
This repository contains the implementation of [DPMLBench: Holistic Evaluation of Differentially Private Machine Learning](https://arxiv.org/abs/2305.05900).

## Requirements
The environment can be set up using [Anaconda](https://www.anaconda.com/download/) with the following commands:

```
conda create --name dpml python=3.8
conda activate dpml
conda install pytorch=1.10.0 pytorch-cuda=11.7 torchvision -c pytorch -c nvidia
pip install -r requirements.txt
```
# Introduction


## Replicate
Run `bash replicate_figure.sh`

## Train

## Attack

## Citation
```

```


## Acknowledgements
Our implementation refers to the source code from the following repositories:
- [DPGEN](https://github.com/tkarras/progressive_growing_of_gans)
- [PrivateSet](https://github.com/DingfanChen/Private-Set)
- [RGP & GEP](https://github.com/jeremy43/Private_kNN)
- [ALIBI](https://github.com/facebookresearch/label_dp_antipodes)
- [PrivateKNN](https://github.com/jeremy43/Private_kNN)
- [Handcraft DP](https://github.com/ftramer/Handcrafted-DP)
- [MI Attack](https://github.com/liuyugeng/ML-Doctor)

