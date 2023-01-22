import pandas as pd
import numpy as np
import argparse
import os
from scipy.ndimage.filters import *

# =================
# === ARGUMENTS ===
# =================

parser = argparse.ArgumentParser(description='Apply Pseudo Action Entropy Threshold on parsed xlsx file')

# File Loading Arguments
parser.add_argument('--csv_boosts_path', required=True, type=str)

# Filtering Arguments
parser.add_argument('--filter', default="gaussian", type=str)
parser.add_argument('--sigma', default=5, type=float, help='variance for gaussian filter')
parser.add_argument('--size', default=15, type=int, help='window size for median filter')

# =========================
# === SUPPORT FUNCTIONS ===
# =========================

def get_filter_filename(args):
    dir = os.path.dirname(args.csv_boosts_path)
    file_no_ext = os.path.splitext(os.path.basename(args.csv_boosts_path))[0]
    if args.filter == 'gaussian':
        file_no_ext += '_gaussian_sigma' + str(args.sigma)
    elif args.filter == 'median':
        file_no_ext += '_median_size' + str(args.size)
    return dir + "\\" + file_no_ext + '.csv'

def apply_filter(args, se_boosts):
    if args.filter == 'gaussian':
        se_filtered = gaussian_filter(input=se_boosts, sigma=args.sigma)
    elif args.filter == 'median':
        se_filtered = median_filter(input=se_boosts, size=args.size)
    else:
        assert False, "Unsupported Filter Argument Type"
    return se_filtered

# =====================
# === MAIN FUNCTION ===
# =====================

def main(args):
    se_boosts = pd.read_csv(args.csv_boosts_path, sep=',',header=None).to_numpy()   # Load Sample Efficiency Boosts
    se_filtered = apply_filter(args, se_boosts) # Apply Filter
    np.savetxt(get_filter_filename(args), se_filtered, delimiter=",") # Save Filtered Boosts

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
