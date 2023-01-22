import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from enum import Enum
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
from typing import Any, List, Sequence, Tuple

from tensorflow.keras import layers
import tensorflow as tf

from src.models import OneLayer, a2c_Model
from src.srl import srl
from src.ac import ac_drl
from src.sb import sb_a2c_drl
from src.config import *

model_dict = {
    "AC-1L" : OneLayer
}

custom_drl_dict = {
    "AC": ac_drl
}

sb_drl_dict = {
    "SB_A2C": sb_a2c_drl
}

drl_dict = {**custom_drl_dict, **sb_drl_dict}

env_names = [
    "CartPole-v0",
    "CartPole-v1"
]

####################################
###    Command Line Arguments    ###
####################################

# ====> Common Parameters
parser = argparse.ArgumentParser(description='Entropy Thresholding Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='AC-1L',
                    choices=model_dict.keys(),
                    help='model architecture: ' + ' | '.join(model_dict.keys()) +
                    ' (default: AC-1L)')
parser.add_argument('--env', '-e', metavar='ENV', default='CartPole-v0',
                    choices=env_names,
                    help='environment names: ' + ' | '.join(env_names) +
                    ' (default: CartPole-v0)')
parser.add_argument('--n_env', '-ne', metavar='ENV', default=1,
                    choices=env_names,
                    help='environment names: ' + ' | '.join(env_names) +
                    ' (default: CartPole-v0)')
parser.add_argument('--hidden_units', metavar='HU', type=int, default=128,
                    help='hidden units (for all layers) (defatul: 128)')
parser.add_argument('--seed', metavar='SD', type=int, default=1000,
                    help='random seed (defatul: 1000)')

# ====> SRL Parameters ('srl_' prefix)
parser.add_argument('--srl_type', default=0, type=int, metavar='SRL',
                    help='whether to do SRL or not (default: 0) ' +
                    '0 - No SRL ' + 
                    '1 - SRL Predicting Next States ' +
                    '2 - SRL Predicting Next State Deltas')
parser.add_argument('--srl_epochs', default=200, type=int, metavar='SRL_N',
                    help='srl epochs (default: 200)')
parser.add_argument('--srl_lr', default=0.01, type=float, metavar='SRL_LR',
                    help='srl learning rate (default: 0.01)')
parser.add_argument('--srl_cos_anneal_period', default=200, type=int, metavar='SRL_CAP',
                    help='srl LR scheduler cosine annealing period (default: 200)')
parser.add_argument('--srl_cos_anneal_mult', default=1, type=int, metavar='SRL_CAM',
                    help='srl LR scheduler cosine annealing multiplier (default: 1)')

# ====> DRL Parameters ('drl_' prefix)
parser.add_argument('--drl_algorithm', default='SB_A2C', metavar='DRL_ALGO', 
                    choices=drl_dict.keys(), help='drl algorithm (default: AC)')
parser.add_argument('--drl_lr', default=0.01, type=float, metavar='DRL_LR',
                    help='drl learning rate (default: 0.01)')
parser.add_argument('--drl_agents', default=100, type=int, metavar='DRL_AG',
                    help='drl agents, for A2C (default: 100)')
parser.add_argument('--drl_gamma', default=0.99, type=float, metavar='DRL_G',
                    help='drl gamma (default: 0.99)')
                    
parser.add_argument('--drl_min_eps_crit', default=constants.DEFAULT, metavar='DRL_MNEC',
                    help='drl consecutive episodes >= reward threshold')
parser.add_argument('--drl_max_eps', default=constants.DEFAULT, metavar='DRL_MXE',
                    help='drl max episodes')
parser.add_argument('--drl_max_steps_per_eps', default=constants.DEFAULT, metavar='DRL_MXSPE',
                    help='drl max steps per episodes')
parser.add_argument('--drl_reward_threshold', default=constants.DEFAULT, metavar='DRL_RT',
                    help='drl reward threshold')

args = parser.parse_args()

env = gym.make(args.env)

###########################
###    MAIN FUNCTION    ###
###########################

def main(args):
    global env

    # Reproducibility
    env.seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    model = model_dict[args.arch](
        num_actions=env.action_space.n, 
        num_hidden_units=args.hidden_units
    )

    # Run SRL
    if args.srl_type != 0:
        srl_optimizer = tf.keras.optimizers.Adam(learning_rate=args.srl_lr)
        srl_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            args.srl_lr, args.srl_epochs, end_learning_rate=0.0001, power=1.0,
            cycle=False, name=None
        )
        srl_loss_func = tf.keras.losses.CategoricalCrossentropy()
        srl(args, model, srl_optimizer, srl_scheduler, srl_loss_func)

    # Custom DRL
    if args.drl_algorithm in custom_drl_dict:
        drl_optimizer = tf.keras.optimizers.Adam(learning_rate=args.drl_lr)
        drl_loss_func = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        drl_dict[args.drl_algorithm](args, model, drl_optimizer, drl_loss_func, env)
    # Stable Baseline DRL
    elif args.drl_algorithm in sb_drl_dict:
        drl_dict[args.drl_algorithm](args)


if __name__ == "__main__":
    main(args)
