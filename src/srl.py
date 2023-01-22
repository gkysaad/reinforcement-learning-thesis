from stable_baselines3_thesis import A2C, DQN, DDPG, PPO, TD3
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import OrderedDict
import pandas as pd
import numpy as np
import argparse
import random
import pickle
import time
import math
import gym
import os
from common import *

parser = argparse.ArgumentParser(description='State Representation Learning (SRL)')

# ==> Env
parser.add_argument('--env', default='CartPole-v0')

# ==> SRL Parameters
parser.add_argument('--dataset_path', default=None, type=str, 
                    help="if path specified, don't generate a dataset and use path's dataset")
parser.add_argument('--dataset_size', default=1000, type=int)
parser.add_argument('--max_iters', default=10000, type=int)
parser.add_argument('--epochs', default=None, type=int, help='If None, train until max_iters')
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--num_neurons_per_layer', default=64, type=int)
parser.add_argument('--traintest_split', default=0.8, type=float)
parser.add_argument('--seed', default=1000, type=int)

# ==> Logging
parser.add_argument('--base_folder', default="C:\\Users\\George\\OneDrive\\Documents\\University\\Y4F\\Thesis\\thesis-rl-project-main\\thesis-rl-project-main\\src\\results\\srl")
parser.add_argument('--dataset_csv', default="srl_dataset.csv", type=str)

# =======================================
# === SRL NEURAL NETOWRK & DATALOADER ===
# =======================================

class srl_network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(srl_network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1)
        return x

class srl_dataLoader(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

# =========================
# === SUPPORT FUNCTIONS ===
# =========================

def set_seed(seed): # IDK WHY BUT REPRODUCIBILITY IS GOOD FOR ALL TRIALS AFTER 1ST (oh well)
    torch.manual_seed(seed)                     # Sets random seed for CPU
    torch.cuda.manual_seed(seed)                # Sets random seed for GPU (Single GPU)
    torch.cuda.manual_seed_all(seed)            # Sets random seed for All GPUs (Multi GPU)
    torch.backends.cudnn.deterministic = True   # Set CUDNN backend to deterministic
    torch.backends.cudnn.benchmark = False      # IDK why, but this helps

def accuracy(output, target):
    '''Compute the percentage of correct bits'''
    output = (output>=0.5) # Round to 1 if value >= 0.5, else 0
    correct = torch.sum((output==target.bool())).item()
    total = output.numel()
    return correct / total

def select_action(model, env, state):
    '''NOTE: For now, we only use random actions, and only support discrete actions'''
    action_space = env.action_space

    if isinstance(action_space, gym.spaces.Discrete):
        if model == None:
            action = random.randint(0, action_space.n-1)
        else:
            action = model(state)
    else:
        assert False, "Non-discrete action space not implemented"
    
    return action

def gen_dataset(args, env, trial_name):
    '''TO DO: Run DRL during dataset generation, and save that model'''

    # Select Agent
    model = None # For now, we don't need a model (random actions)
    
    # Generate Dataset
    prev_states, next_states, actions, rewards = [], [], [], [] # Environment variables logging
    num_samples = 0
    state = env.reset()
    print("==> Start Generating Dataset")
    tic = time.time()
    while num_samples < args.dataset_size:
        prev_states.append(state)

        # Perform an action in environment
        action = select_action(model, env, state)
        state, reward, done, _ = env.step(action)

        # Logging
        next_states.append(state)
        rewards.append(reward)
        actions.append(action)

        if done == True:
            state = env.reset()
        num_samples += 1

    toc = time.time()
    print("==> Complete Dataset Generation In ", toc-tic, "s")

    # Save to CSV
    file_path = args.base_folder + "\\" + trial_name + "\\" + args.dataset_csv
    df = pd.DataFrame(data={
        "prev_states": prev_states, 
        "actions": actions, 
        "next_states": next_states, 
        "rewards": rewards})

    df.to_csv(file_path, index=False)
    print("==> Saved Dataset To: ", file_path)
    return file_path

def series_to_list(df_series):
    arr = []
    for state in df_series.tolist():
        state = state.replace('[', '').replace(']', '')
        state = list(state.split())
        state = list(map(float,state))
        arr.append(state)
    return arr

def get_data_and_labels(file_path, nrows=None):
    df = pd.read_csv(file_path, nrows=nrows)

    # Data Preprocessing
    prev_states = series_to_list(df["prev_states"])
    next_states = series_to_list(df["next_states"])
    actions = df["actions"].tolist()

    data = [p_state + [act] for p_state, act in zip(prev_states, actions)]
    data = torch.FloatTensor(data)
    labels = next_states
    labels = torch.FloatTensor(labels)

    return data, labels

def get_sb3_model_wandb_dict(env, srl_model):
    '''Returns ordered dictionary required for stable baselines3 DRL model'''
    state_dim = env.observation_space._shape[0]
    with torch.no_grad():
        copy_dims = [
            [None, None, None, state_dim],
            [None, None],
            [None, None, None, None],
            [None, None],
            -1,
            -1,
        ]
        srl_params = []    
        for idx, srl_param in enumerate(srl_model.parameters()):
            dims = copy_dims[idx]
            if copy_dims[idx] == -1:
                continue
            elif len(copy_dims[idx]) == 4:
                srl_params.append(srl_param.data[dims[0]:dims[1], dims[2]:dims[3]].cuda())
            elif len(copy_dims[idx]) == 2:
                srl_params.append(srl_param.data[dims[0]:dims[1]].cuda())
        
        drl_new_params = {}
        drl_new_params['policy'] = OrderedDict()
        drl_new_params['policy']['mlp_extractor.policy_net.0.weight'] = srl_params[0]
        drl_new_params['policy']['mlp_extractor.policy_net.0.bias'] = srl_params[1]
        drl_new_params['policy']['mlp_extractor.policy_net.2.weight'] = srl_params[2]
        drl_new_params['policy']['mlp_extractor.policy_net.2.bias'] = srl_params[3]
        
    return drl_new_params

# ======================
# === MAIN FUNCTIONS ===
# ======================

def main(args):
    set_seed(args.seed)

    # ==> Trial Name
    num_epochs = args.epochs if args.epochs != None else math.ceil(args.max_iters/args.dataset_size)
    trial_name = args.env + "_DatasetSize" + str(args.dataset_size) + "_Epochs" + str(num_epochs) + "_lr" + str(args.lr) + "_neurons" + str(args.num_neurons_per_layer) + "_seed" + str(args.seed)

    # ==> Load and/or create dataset
    assert os.path.exists(args.base_folder + "\\" + trial_name) == False, "SRL Trial Directory Exists"
    os.mkdir(args.base_folder + "\\" + trial_name)
    env = gym.make(args.env)
    if args.dataset_path == None:
        file_path = gen_dataset(args, env, trial_name)
        data, labels = get_data_and_labels(file_path)
    else:
        data, labels = get_data_and_labels(args.dataset_path)
    
    trainset = srl_dataLoader(data[:int(len(data) * args.traintest_split)], labels[:int(len(labels) * args.traintest_split)])
    testset = srl_dataLoader(data[int(len(data) * args.traintest_split):], labels[int(len(labels) * args.traintest_split):])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                               shuffle=False, pin_memory=True)

    # ==> Run SRL Training
    state_dim = env.observation_space._shape[0]
    action_dim = env.action_space.n
    model = srl_network(input_size=state_dim + 1,
                        output_size=state_dim,
                        hidden_size=args.num_neurons_per_layer).cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()

    # Data Logging Variables
    train_loss, train_time = [], []
    test_loss, test_time = [], []

    # Training Loop
    start_tic = time.time()
    iters = 0
    for epoch in range(num_epochs):
        if iters >= args.max_iters:
            break
        
        # Training
        sub_train_loss = []
        tic = time.time()

        model.train()
        for input, target in train_loader:
            output = model(input.cuda()).cuda()
            loss = criterion(output.cuda(), target.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sub_train_loss.append(loss.float().item() / input.size(0))
            iters += input.size(0)

        train_loss.append(sum(sub_train_loss)/len(sub_train_loss))
        train_time.append(time.time() - tic)

        # Testing
        with torch.no_grad():
            sub_test_loss = []
            tic = time.time()

            model.eval()
            for input, target in test_loader:
                output = model(input.cuda())
                loss = criterion(output.cuda(), target.cuda())

                sub_test_loss.append(loss.float().item() / input.size(0))

            test_loss.append(sum(sub_test_loss)/len(sub_test_loss))
            test_time.append(time.time() - tic)

        print('EPOCH: {0}\n'
            'Train | time:{1:.3f}\tLoss: {2:.7f}\n'
            'Test  | time:{3:.3f}\tLoss: {4:.7f}'
            .format(epoch+1, 
                train_time[epoch], train_loss[epoch], # Train
                test_time[epoch], test_loss[epoch])) # Test
        print('-----------')

    print("======> Total Time: ", time.time() - start_tic)

    # Logging Data
    df1 = pd.DataFrame()
    df1["train_loss"] = train_loss
    df1["test_loss"] = test_loss
    df1["train_time"] = train_time
    df1["test_time"] = test_time
    df1.insert(loc=0, column='epoch', value=np.arange(len(df1)))

    # Model And Summary Data
    data = [
        ["num neurons per layer", args.num_neurons_per_layer],
        ["lr", args.lr],
        ["seed", args.seed],
        ["split training data", args.traintest_split],
        ["min train loss", min(train_loss)],
        ["min test loss", min(test_loss)],
        ["end train loss", train_loss[epoch]],
        ["end test loss", test_loss[epoch]],
    ]
    df2 = pd.DataFrame(data)

    # Create a Pandas Excel writer
    xlsx_file = args.base_folder + "\\" + trial_name + "\\" + "srl_training_log.xlsx"
    writer = pd.ExcelWriter(xlsx_file, engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    df1.to_excel(writer, sheet_name='Logged Data', index = False)
    df2.to_excel(writer, sheet_name='Model And Summary Data', index = False, header = False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    # ==> Save DRL parameters dictionary
    sb3_param_dict = get_sb3_model_wandb_dict(env, model)
    sb3_param_filename = args.base_folder + "\\" + trial_name + "\\" + "drl_param_dict.pkl"
    sb3_param_file = open(sb3_param_filename, "wb")
    pickle.dump(sb3_param_dict, sb3_param_file)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    # file_path = "C:\\Users\\Michael Ruan\\Documents\\Github\\thesis-rl-project\\src\\data\\srl_datasets\\CartPole-v0.csv"
    # get_data_and_labels(file_path, nrows=10)
