import pandas as pd
import os

# loop through activations folder
act_folder = ".activations/"
act_files = os.listdir(act_folder)
act_files = [act_folder + f for f in act_files if f.endswith(".csv")]

# create dataframe
df = pd.DataFrame(columns=["num_iterations", "layer1", "layer2"])

# loop through files
for f in act_files:
    # read in file
    df_temp = pd.read_csv(f, header=None)
    # get num rows
    num_rows = df_temp.shape[0]
    # get average of columns
    df_temp = df_temp.mean(axis=0)
    # add to dataframe
    df = df.append({"num_iterations": num_rows, "layer1": df_temp[0], "layer2": df_temp[1]}, ignore_index=True)

df.plot(y="num_iterations", x=["layer1", "layer2"], kind="scatter")

