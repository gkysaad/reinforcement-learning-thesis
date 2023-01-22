from enum import Enum

class constants(Enum):
    DEFAULT = -1

cartpole_v0_params = {
    "drl_min_eps_crit": 100,
    "drl_max_eps": 10000,
    "drl_max_steps_per_eps": 200,
    "drl_reward_threshold": 195
}

cartpole_v1_params = {
    "drl_min_eps_crit": 100,
    "drl_max_eps": 10000,
    "drl_max_steps_per_eps": 500,
    "drl_reward_threshold": 475
}
