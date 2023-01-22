#!/bin/bash

python sb.py --model A2C --policy MlpPolicy --repeat 100 --seed 800 --project final_n_env_10_arch_064_064_lr_2.50e-3_seed_0800 --lr 2.50e-3
python sb.py --model A2C --policy MlpPolicy --repeat 100 --seed 900 --project final_n_env_10_arch_064_064_lr_2.50e-3_seed_0900 --lr 2.50e-3
