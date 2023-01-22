#!/bin/bash

python sb.py --model A2C --policy MlpPolicy --n_env 10 --env CartPole-v0 --ent_coef 0.02 --repeat 100 --reward_threshold 195 --steps 100000 --project srl_1000samples_100srlepochs_n_env_10_arch_096_096_lr_7.50e-4 --lr 7.50e-4 --num_neurons_per_layer 96 --weights_path "C:\\Users\\George\\OneDrive\\Documents\\University\\Y4F\\Thesis\\thesis-rl-project-main\\thesis-rl-project-main\\src\\results\\srl\\CartPole-v0_DatasetSize1000_Epochs100_lr0.001_neurons96_seed1000\\drl_param_dict.pkl"
python sb.py --model A2C --policy MlpPolicy --n_env 10 --env CartPole-v0 --ent_coef 0.02 --repeat 100 --reward_threshold 195 --steps 100000 --project srl_1000samples_100srlepochs_n_env_10_arch_096_096_lr_1.00e-3 --lr 1.00e-3 --num_neurons_per_layer 96 --weights_path "C:\\Users\\George\\OneDrive\\Documents\\University\\Y4F\\Thesis\\thesis-rl-project-main\\thesis-rl-project-main\\src\\results\\srl\\CartPole-v0_DatasetSize1000_Epochs100_lr0.001_neurons96_seed1000\\drl_param_dict.pkl"
python sb.py --model A2C --policy MlpPolicy --n_env 10 --env CartPole-v0 --ent_coef 0.02 --repeat 100 --reward_threshold 195 --steps 100000 --project srl_1000samples_100srlepochs_n_env_10_arch_096_096_lr_2.50e-3 --lr 2.50e-3 --num_neurons_per_layer 96 --weights_path "C:\\Users\\George\\OneDrive\\Documents\\University\\Y4F\\Thesis\\thesis-rl-project-main\\thesis-rl-project-main\\src\\results\\srl\\CartPole-v0_DatasetSize1000_Epochs100_lr0.001_neurons96_seed1000\\drl_param_dict.pkl"
python sb.py --model A2C --policy MlpPolicy --n_env 10 --env CartPole-v0 --ent_coef 0.02 --repeat 100 --reward_threshold 195 --steps 100000 --project srl_1000samples_100srlepochs_n_env_10_arch_096_096_lr_5.00e-3 --lr 5.00e-3 --num_neurons_per_layer 96 --weights_path "C:\\Users\\George\\OneDrive\\Documents\\University\\Y4F\\Thesis\\thesis-rl-project-main\\thesis-rl-project-main\\src\\results\\srl\\CartPole-v0_DatasetSize1000_Epochs100_lr0.001_neurons96_seed1000\\drl_param_dict.pkl"
