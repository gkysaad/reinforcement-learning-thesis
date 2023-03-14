from tkinter import N
import torch
import argparse
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import entropy, norm
import seaborn as sb
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import csv
from tqdm import tqdm

from stable_baselines3_thesis.common.torch_layers import MlpExtractor
from captum.attr import LayerGradCam, GuidedGradCam

# =================
# === ARGUMENTS ===
# =================

parser = argparse.ArgumentParser(description='Analysing neuron activations\
     in hidden layers')

# ==> Logging Parameters
parser.add_argument('--dir', required=True)
parser.add_argument('--xlsx_name', default=None)
parser.add_argument('--heatmap_name', default="heatmap.png")
parser.add_argument('--heatmap_percent_name', default="heatmap_pc.png")

# ==> Env Parameters
parser.add_argument('--num_actions', default=1)
parser.add_argument('--n_eval', default=50)
parser.add_argument('--n_env_eval', default=1000)
parser.add_argument('--n_steps_max', default=100000)
parser.add_argument('--reward_threshold', default=195, type=int)
parser.add_argument('--method', default="gradcam")
parser.add_argument('--act_fcn', default="tanh")
parser.add_argument('--env', default="CartPole-v0")

# ==> Threshold Parameters
parser.add_argument('--task', default='neuron_entropy_grad')
parser.add_argument('--threshold_steps', default=11, type=int)
parser.add_argument('--grad_threshold_steps', default=4, type=int)
parser.add_argument('--weight_grad', default=0, type=int, help='whether to weight\
     neuron activation entropy by grad magnitude')
parser.add_argument('--run_type', default='train', help='train or test')
parser.add_argument('--threshold', default=None, type=float)


def get_tensors_and_model(args):
    files_actvs = []
    files_feats = []
    files_model = []

    run_dirs = [os.path.join(args.dir, directory) for directory in \
        os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, directory))]
    run_dirs = [(path, int(path.split("_")[-1])) for path in run_dirs]
    run_dirs = sorted(run_dirs, key=lambda x: x[1])
    run_dirs = [dir for dir, _ in run_dirs]

    # Parse Directories
    for dir in run_dirs:
        paths = os.listdir(dir)

        ordered_paths_actvs = [os.path.splitext(path)[0] for path in paths\
             if path.endswith('.pt') and path.startswith('atv_and_act')]
        ordered_paths_actvs = [(path, int(path.split("_")[-1])) for path in\
             ordered_paths_actvs]
        ordered_paths_actvs = sorted(ordered_paths_actvs, key=lambda x: x[1])

        ordered_paths_feats = [os.path.splitext(path)[0] for path in paths\
             if path.endswith('.pt') and path.startswith('features')]
        ordered_paths_feats = [(path, int(path.split("_")[-1])) for path in\
             ordered_paths_feats]
        ordered_paths_feats = sorted(ordered_paths_feats, key=lambda x: x[1])

        ordered_paths_model = [os.path.splitext(path)[0] for path in paths\
             if path.endswith('.pt') and path.startswith('model')]
        ordered_paths_model = [(path, int(path.split("_")[-1])) for path\
             in ordered_paths_model]
        ordered_paths_model = sorted(ordered_paths_model, key=lambda x: x[1])

        files_actvs.append([os.path.join(dir, file+".pt")\
            for file, _ in ordered_paths_actvs])
        files_feats.append([os.path.join(dir, file+".pt")\
            for file, _ in ordered_paths_feats])
        files_model.append([os.path.join(dir, file+".pt")\
            for file, _ in ordered_paths_model])

        with open('model.json', 'w') as f:
            json.dump(files_model, f)

    # Load Activation Tensors and Stack
    atv_tensors = []
    act_tensors = []
    feats_tensors = []
    models = []

    for actvs_run, feats_run, model_run in zip(files_actvs, \
        files_feats, files_model):
        sub_atv_tensors = []
        sub_act_tensors = []
        sub_feats_tensors = []
        sub_models = []
        for actvs_path, feats_path, model_path in \
            zip(actvs_run, feats_run, model_run):
            atv_act_tensor = torch.load(actvs_path).cpu()
            sub_atv_tensors.append(atv_act_tensor[:, :128])
            sub_act_tensors.append(atv_act_tensor[:, 128:])
            sub_feats_tensors.append(torch.load(feats_path).cpu())
            if args.act_fcn == "tanh":
                model = MlpExtractor(feature_dim=4, \
                    net_arch=[dict(pi=[64,64], vf=[64,64])], activation_fn=nn.Tanh).cpu()
            elif args.act_fcn == "relu":
                model = MlpExtractor(feature_dim=4, \
                    net_arch=[dict(pi=[64,64], vf=[64,64])], activation_fn=nn.ReLU).cpu()
            model.load_state_dict(torch.load(model_path))
            model.gradcam_forward = True
            sub_models.append(model)

        act_tensors.append(torch.vstack(sub_act_tensors))
        atv_tensors.append(torch.vstack(sub_atv_tensors))
        feats_tensors.append(torch.vstack(sub_feats_tensors))
        models.append(sub_models)
    
    return models, atv_tensors, act_tensors, feats_tensors

def get_out_file_name(dir, out_xlsx):
    if out_xlsx == None:
        project_name = os.path.normpath(dir).split(os.path.sep)[-1]# remove extension
        out_xlsx = project_name + ".xlsx"
    return out_xlsx

def load_sheets(args, dir, xlsx_name):
    # Excel data source
    source_file = dir + "\\" + xlsx_name

    # Load and Process Eval Reward Sheet
    df_rew = pd.read_excel(open(source_file, 'rb'), sheet_name='eval mean_reward')
    df_rew = df_rew.loc[:,~df_rew.columns.str.match("Unnamed")] # Drop unnamed columns
    df_rew = df_rew.drop(columns=['step'])                      # Drop step column
    df_rew[df_rew < args.reward_threshold] = 0

    df_rew_step = []
    for col in df_rew.columns:
        sol_idx = -1
        for idx, rew in df_rew[col].items():
            if float(rew) > 0:
                sol_idx = idx
                break
        df_rew_step.append((sol_idx+1)*args.n_env_eval if sol_idx != -1 else sol_idx)
    return df_rew_step

# def action_entropy(args, actions, rew_step, threshold):
#     sample_efficiency = 0
#     num_success = 0
#     num_iters = 0
#     for i in range(len(rew_step)):
#         action_entropy = entropy
#     return sample_efficiency

def get_neuron_activation_with_grads(feats, model, actions, activations):
    # Calculate neuron entropy per run
    # Neuron activation entropy calculated w.r.t. each action, for each neuron
    # action_values = [0,1]
    atv_with_grad = [] # [run1, run2, ...] -> run1={action1:((atv1, grad1), ...
    counter = 0
    for f, m, a, av in tqdm(zip(feats, model, actions, activations), total = min(len(feats), len(model), len(actions), len(activations))):
        counter += 1
        action_run_atv_with_grad = {}
        # for act in action_values:
        #     act_idx = torch.from_numpy(np.where(a == act)[0])
            # act_activations = av[act_idx, :]
        act_activations = av
        sub_run_atv_with_grad = []
        # print("act_activations: ", act_activations.shape, " act: ", act, " counter: ", counter)
        for idx in range(act_activations.shape[1]):
            neuron_activation = act_activations[:, idx].detach().numpy()\
                .squeeze()
            if args.method == 'gradcam':
                layer = LayerGradCam(m, m.policy_net[0] \
                    if idx < 64 else m.policy_net[2])
            elif args.method == 'gbp':
                layer = GuidedGradCam(m, m.policy_net[0] \
                    if idx < 64 else m.policy_net[2])
            
            grad = layer.attribute(f, target=idx%64).squeeze()\
                .detach().numpy()
            sub_run_atv_with_grad.append((neuron_activation, grad))
        # action_run_atv_with_grad[act] = sub_run_atv_with_grad
        action_run_atv_with_grad["all"] = sub_run_atv_with_grad
        atv_with_grad.append(action_run_atv_with_grad)

    return atv_with_grad

def get_neuron_entropy(actions, activations):
    # Calculate neuron entropy per run
    # Neuron activation entropy calculated w.r.t. each action, for each neuron
    print("enters get_neuron_entropy")
    action_values = [0,1]
    run_entropy = []
    for a, av in zip(actions, activations):
        neuron_entropy = 0
        for act in action_values:
            act_idx = torch.from_numpy(np.where(a == act)[0])
            act_activations = av[act_idx, :]
            for idx in range(act_activations.shape[1]):
                neuron_entropy += entropy(norm.pdf(\
                    act_activations[:, idx].detach().numpy().squeeze()))
        run_entropy.append(neuron_entropy)
    return run_entropy

def neuron_entropy(args, actions, activations, rew_step, threshold):
    # Calculate the sample efficiency boost for some threshold percent
    print("enters neuron_entropy")
    run_entropy = get_neuron_entropy(actions, activations)
    threshold_idx = int(threshold*len(run_entropy)) if threshold != 1 \
        else len(run_entropy)-1
    threshold_val = sorted(run_entropy)[threshold_idx]
    total_iters = 0
    total_success = 0
    for idx, step in enumerate(rew_step):
        if run_entropy[idx] < threshold_val:
            if step != -1:
                total_success += 1
                total_iters += step
            else:
                total_iters += args.n_steps_max
        else:
            total_iters += args.n_eval
    sample_efficiency = total_success / total_iters
    return sample_efficiency

def neuron_entropy_with_grad(args, run_activations, rew_step,\
     neuron_threshold, grad_threshold, weight_grad=0):
    # print("weight_grad: ", weight_grad)
    # Calculate Neuron Entropy
    run_with_grad_entropy = []
    print("num runs: ", len(run_activations))
    for run in run_activations:
        run_ent_val = 0
        # print("run: ", run)
        # print("run: ", run.items())
        for _, act_and_grad in run.items():
            activation = torch.tensor([i[0] for i in act_and_grad])
            gradients = torch.tensor([i[1] for i in act_and_grad])

            # print("activation: ", activation.shape)
            # print("gradients: ", gradients.shape)

            if weight_grad == 1:
                # print("GRAD WEIGHTED ENTROPY")
                activation = activation * gradients
            # k = min(int(grad_threshold*128), 127) 
            # grad_mags, topk_idxs = torch.topk(grad_mean, k) # just take neurons with largest grads
            # if len(topk_idxs) != 0:
                # for grad_mag, idx in zip(grad_mags, topk_idxs): # idx = neurons
                #     if weight_grad == 0:
            # print(activation[idx, :].detach().numpy().squeeze())
            # print(activation.detach().numpy().squeeze().shape)
            # hist_activations = np.histogram(activation[idx, :].detach().numpy().squeeze(), bins=7)
            hist_activations = np.histogram(activation.detach().numpy().squeeze(), bins=7)
            hist_counts = hist_activations[0]
            hist_freqs = hist_counts / np.sum(hist_counts)
            run_ent_val = -np.sum(hist_freqs * np.log(hist_freqs, out=np.zeros_like(hist_freqs), where=(hist_freqs!=0)))
                        # run_ent_val += entropy(hist_freqs) 
                        # run_ent_val += entropy(norm.pdf(\
                        #     activation[idx, :].detach().numpy().squeeze()))
                        # threshold with median
                        # standard deviation
                        # put into buckets (7 bins) then compute entropy
                    # elif weight_grad == 1:
                    #     run_ent_val += grad_mag.item()*entropy(\
                    #         norm.pdf(activation[idx, :].detach().numpy()\
                    #             .squeeze()))
        run_with_grad_entropy.append(run_ent_val)
    
    # Calculate the sample efficiency boost for some threshold percent
    # threshold_idx = int(neuron_threshold*len(run_entropy)) if \
    #     neuron_threshold != 1 else len(run_entropy)-1
    # threshold_val = sorted(run_with_grad_entropy)[threshold_idx]
    threshold_val = neuron_threshold
    total_iters = 0
    total_success = 0
    # plt.scatter(range(len(run_with_grad_entropy)), run_with_grad_entropy)
    # plt.show()
    for idx, step in enumerate(rew_step):
        # print("step: ", step, " idx: ", idx, " run_with_grad_entropy: ", run_with_grad_entropy[idx], " threshold_val: ", threshold_val)
        if run_with_grad_entropy[idx] > threshold_val:
            if step != -1:
                total_success += 1
                total_iters += step
            else:
                total_iters += args.n_steps_max
        else:
            total_iters += args.n_eval # not sure what n_eval is
    sample_efficiency = total_success / total_iters
    print("Sample Efficiency: ", sample_efficiency)
    return sample_efficiency

def main(args):
    models, activations, actions, feats = get_tensors_and_model(args)
    xlsx_name = get_out_file_name(args.dir, args.xlsx_name)

    # gets the # of steps at which the reward threshold is reached for each run
    rew_step = load_sheets(args, args.dir, xlsx_name)

    print("models: ", len(models), " activations: ", len(activations), " actions: ", len(actions), " feats: ", len(feats), " rew_step: ", len(rew_step))

    start_ind = int(args.dir[-3])
    for i in range(1,5):
        i += start_ind
        args.dir = args.dir[:-3] + str(i*100)
        args.xlsx_name = args.xlsx_name[:-8] + str(i*100) + ".xlsx"
        print("args.xlsx_name: ", args.xlsx_name)
        models2, activations2, actions2, feats2 = get_tensors_and_model(args)

        models = models + models2
        activations = activations + activations2
        actions = actions + actions2
        feats = feats + feats2

        xlsx_name = get_out_file_name(args.dir, args.xlsx_name)
        rew_step2 = load_sheets(args, args.dir, xlsx_name)
        rew_step = rew_step + rew_step2 

    print("models: ", len(models), " activations: ", len(activations), " actions: ", len(actions), " feats: ", len(feats), " rew_step: ", len(rew_step))

    first_models = [m[1] for m in models]
    first_feats = [f[50:args.n_eval+50, :] for f in feats]
    first_actions = [a[50:args.n_eval+50, :] for a in actions]
    first_activations = [a[50:args.n_eval+50, :] for a in activations]

    # print(first_activations[0].shape)
    # print(first_actions[0].shape)

    if args.task == "neuron_entropy":
        # ignore, not run
        print("THIS SHOULD NOT RUN")
        ne_boosts = []
        thresholds = np.linspace(0, 1, args.threshold_steps)
        for threshold in thresholds:
            ne = neuron_entropy(args, first_actions, first_activations, \
                rew_step, threshold)
            ne_boosts.append(ne)

        plt.plot(thresholds, ne_boosts)
        plt.show()
    elif args.task == "neuron_entropy_grad":
        print("ENTERS CORRECT TASK")
        atv_with_grad = get_neuron_activation_with_grads(first_feats, 
                                                        first_models, 
                                                        first_actions, 
                                                        first_activations)
        # run_entropy = get_neuron_entropy(first_actions, first_activations)

        # print("run_entropy shape: ", len(run_entropy))
        # print("run_entropy: ", run_entropy)
        if args.weight_grad == 0:
            if args.act_fcn == "relu":
                thresholds = np.linspace(0, 1, args.threshold_steps)
                thresholds = np.delete(thresholds, 0)
            elif args.act_fcn == "tanh":
                if args.env == "CartPole-v1":
                    thresholds = np.linspace(0.6, 1.6, args.threshold_steps)
                else:
                    thresholds = np.linspace(1, 2, args.threshold_steps)
        else:
            thresholds = np.linspace(0.2, 1.4, args.threshold_steps)
        thresholds = np.insert(thresholds, 0, 0)
        # thresholds = np.around(np.linspace(0, 1, args.threshold_steps)\
        #     , decimals=1)
        # grad_thresholds = np.around(np.linspace(\
        #     0, 1, args.grad_threshold_steps), decimals=1)
        se_boosts = np.zeros((len(thresholds)))
        for n_idx, n_threshold in enumerate(thresholds):
            # for ng_idx, ng_threshold in enumerate(grad_thresholds):
            # se_boosts[ng_idx, n_idx] = neuron_entropy_with_grad(args, \
            #     atv_with_grad, run_entropy, rew_step, n_threshold, \
            #     ng_threshold, args.weight_grad)
            se_boosts[n_idx] = neuron_entropy_with_grad(args, \
                atv_with_grad, rew_step, n_threshold, \
                0, args.weight_grad)
            # print("PROGRESS: ", n_idx)
            print("PROGRESS: ", n_idx, " ", se_boosts[n_idx])
        
        print(se_boosts)
        print("Best Threshold: ", thresholds[np.argmax(se_boosts)])

        print("SE Boost Percents: ")
        percents = [(se_boosts[i] - se_boosts[0])/se_boosts[0] * 100 for i in range(len(se_boosts))]
        print(percents)

        # sb.heatmap(se_boosts, xticklabels=thresholds,\
        #      yticklabels=grad_thresholds)
        plt.plot(thresholds, se_boosts, label="Sample Efficiency", color="blue")
        plt.xlabel("Neuron Entropy Threshold")
        plt.ylabel("Sample Efficiency")
        plt.plot(thresholds[np.argmax(se_boosts)], np.max(se_boosts), marker="o", color="red")
        plt.tight_layout()
        if args.weight_grad == 0:
            plt.savefig("neuron_entropy_{}_{}_{}_{}.png".format(args.run_type, len(thresholds), args.act_fcn, args.env[-2:]))
        else:
            plt.savefig("neuron_entropy_grad_{}_{}_{}_{}.png".format(args.run_type, len(thresholds), args.act_fcn, args.env[-2:]))
        plt.show()

        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

        # create csv file to store results
        if args.weight_grad == 0:
            csv_file = open(f"{args.run_type}_results_{len(thresholds)}_{args.act_fcn}_{args.env[-2:]}.csv", "w", newline="")
        else:
            csv_file = open(f"{args.run_type}_grad_results_{len(thresholds)}_{args.act_fcn}.csv", "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Threshold", "Sample Efficiency", "Percent Change"])
        for i in range(thresholds.shape[0]):
            csv_writer.writerow([thresholds[i], se_boosts[i], percents[i]])
        csv_file.close()

        # pc_boosts = se_boosts/se_boosts[-1, -1]
        # nan_pc_boosts = pc_boosts.copy()
        # nan_pc_boosts[nan_pc_boosts == 0] = np.nan
        # print("AVG BOOST:", np.nanmean(nan_pc_boosts))
        # print("MAX BOOST: ", np.amax(pc_boosts))
        # print("VAL BOOST 0.7: ", pc_boosts[-1, int(pc_boosts.shape[1]*0.7)])
        # print("VAL BOOST 0.8: ", pc_boosts[-1, int(pc_boosts.shape[1]*0.8)])

        # results_file = open(os.path.join(args.dir, "results.txt"), "w")
        # results_file.write("AVG BOOST: " + str(np.nanmean(nan_pc_boosts)) + "\n" + \
        #     "MAX BOOST: " + str(np.amax(pc_boosts)) + "\n" + \
        #     "VAL BOOST 0.7: " + str(pc_boosts[-1, int(pc_boosts.shape[1]*0.7)]) + \
        #     "\n" + "VAL BOOST 0.8: " + str(pc_boosts[-1, int(pc_boosts.shape[1]*0.8)]))
        # results_file.close()

        # # sb.heatmap(pc_boosts, xticklabels=thresholds, yticklabels=grad_thresholds)
        # sb.heatmap(pc_boosts, xticklabels=thresholds)
        # plt.tight_layout()
        # # print(os.path.exists(args.dir))
        # plt.savefig(os.path.join(args.dir, "heatmap_pc.png"))

        # plt.figure().clear()
        # plt.close()
        # plt.cla()
        # plt.clf()

        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

