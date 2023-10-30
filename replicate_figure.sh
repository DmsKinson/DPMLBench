

python3 plot_scripts/ploter_static.py -a --datasets mnist cifar10 --format pdf --funcs model_design --name acc_model_design
python3 plot_scripts/ploter_static.py -a --datasets mnist cifar10 --format pdf --funcs model_train --name acc_model_train
python3 plot_scripts/ploter_static.py -a --datasets mnist cifar10 --format pdf --funcs model_ensemble --name acc_model_ensemble
python3 plot_scripts/ploter_static.py -a --datasets mnist cifar10 --format pdf --funcs data_preparation --name acc_data_preparation
python3 plot_scripts/ploter_static.py -a --datasets mnist cifar10 --format pdf --funcs label_cmp --name acc_label_cmp
python3 plot_scripts/ploter_static.py -a --datasets fmnist svhn --format pdf --metric auc_norm --funcs model_ensemble --name acc_model_ensemble_apx
python3 plot_scripts/ploter_static.py -a --datasets fmnist svhn --format pdf --metric auc_norm --funcs model_train --name acc_model_train_apx
python3 plot_scripts/ploter_static.py -a --datasets fmnist svhn --format pdf --metric auc_norm --funcs model_design --name acc_model_design_apx
python3 plot_scripts/ploter_static.py -a --datasets fmnist svhn --format pdf --metric auc_norm --funcs data_preparation --name acc_data_preparation_apx
python3 plot_scripts/ploter_static.py --multi --dataset cifar10 --format pdf --metric auc_norm --funcs data_preparation --name multi_mia_data_preparation
python3 plot_scripts/ploter_static.py --multi --dataset cifar10 --format pdf --metric auc_norm --funcs model_design --name multi_mia_model_design
python3 plot_scripts/ploter_static.py --multi --dataset cifar10 --format pdf --metric auc_norm --funcs model_train --name multi_mia_model_train
python3 plot_scripts/ploter_static.py --multi --dataset cifar10 --format pdf --metric auc_norm --funcs model_ensemble --name multi_mia_model_ensemble
python3 plot_scripts/ploter_static.py --multi --dataset cifar10 --format pdf --metric auc_norm --funcs label_cmp --name multi_mia_label_cmp
