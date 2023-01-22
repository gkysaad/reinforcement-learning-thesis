#!/bin/bash

python sb.py --model A2C --policy MlpPolicy --repeat 100 --seed 600 --project final_n_env_10_arch_064_064_lr_2.50e-3_seed_0600 --lr 2.50e-3
python sb.py --model A2C --policy MlpPolicy --repeat 100 --seed 700 --project final_n_env_10_arch_064_064_lr_2.50e-3_seed_0700 --lr 2.50e-3