#!/bin/bash

python sb.py --model A2C --policy MlpPolicy --repeat 100 --seed 200 --project final_n_env_10_arch_064_064_lr_2.50e-3_seed_0200 --lr 2.50e-3
python sb.py --model A2C --policy MlpPolicy --repeat 100 --seed 300 --project final_n_env_10_arch_064_064_lr_2.50e-3_seed_0300 --lr 2.50e-3