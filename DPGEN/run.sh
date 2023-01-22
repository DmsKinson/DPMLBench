# +
export EXP_NAME=RANDOM_0116_FMNIST_EXP_EXP10_K256_IMAGE_SIZE_28_BATCH_128
export CUDA_VISIBLE_DEVICES=0,1; python main.py --config fmnist.yml --doc $EXP_NAME --exp exp


