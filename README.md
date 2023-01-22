# thesis-rl-project
Decrease training iterations in DRL with entropy threshold

# Commands
 - CartPole-v0
     - python main.py --arch AC-1L --drl_min_eps_crit 100 --drl_max_eps 10000 --drl_max_steps_per_eps 1000 --drl_reward_threshold 195 --drl_gamma 0.99
 - Tensorboard (Command Line)
     - tensorboard --logdir "C:\Users\George\OneDrive\Documents\University\Y4F\Thesis\thesis-rl-project-main\thesis-rl-project-main\src\results"

# Models
 - Actions Spaces:
     - spaces.Box
         - A2C, DDPG, TD3, SAC, PPO
     - spaces.Discrete
         - A2C, PPO, DQN
     - spaces.MultiDiscrete
         - A2C, PPO
     - spaces.MultiBinary
         - A2C, PPO
 - Entropy Logging:
     - A2C, PPO, SAC*(minor change)
 - Replay Buffer:
     - DDPG, DQN, SAC, TD3