import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


# loop through activations folder
act_folder = ".activations/"
act_files = os.listdir(act_folder)
act_files = [act_folder + f for f in act_files if f.endswith(".csv")]

hist_dir = ".histograms/"

chunks = 2000

iterations = np.arange(0, 50000, chunks)
lists = [[] for i in iterations]

# create dataframe
df = pd.DataFrame(columns=["num_iterations", "layer1", "layer2"])

def add_to_lists(x):
    lists[x.index[0]//chunks] += x[1].values.tolist()
    

# loop through files
for i,f in enumerate(act_files):
    # read in file
    df_temp = pd.read_csv(f, header=None)
    df_temp.groupby(df_temp.index // chunks).apply(add_to_lists)

for i, vals in enumerate(lists):
    plt.hist(vals, bins=8, range=[0.9, 2.0])
    plt.title("Histogram of Entropy Values for Iteration " + str(i*chunks))
    plt.xlabel("Entropy Value")
    plt.ylabel("Frequency")
    plt.savefig(hist_dir + "hist_" + str(i*chunks) + ".png")
    print("saved hist_" + str(i*chunks) + ".png")
    plt.clf()

# print(df.head(), df.shape)
# df.plot(y="num_iterations", x="layer1", kind="scatter")
# df.plot(y="num_iterations", x="layer2", kind="scatter")

# plt.show()