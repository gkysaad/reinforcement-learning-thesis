#!/bin/bash

for ((i=0;i<=9;i++)); 
do 
   python neuron_entropy_grad.py --dir "C:\Users\George\OneDrive\Documents\University\Y4F\Thesis\thesis-rl-project-main\thesis-rl-project-main\src\results-relu\drl\CartPole-v0\final_n_env_10_arch_064_064_lr_2.50e-3_seed_0${i}00" \
    --xlsx_name "final_n_env_10_arch_064_064_lr_2.50e-3_seed_0${i}00.xlsx" --act_fcn "relu" --method "gradcam" \
    --heatmap_name "final_n_env_10_arch_064_064_lr_2.50e-3_seed_0${i}00_gradcam_heatmap.png" \
    --heatmap_percent_name "final_n_env_10_arch_064_064_lr_2.50e-3_seed_0${i}00_gradcam_heatmap_percent.png"
   echo $i
done