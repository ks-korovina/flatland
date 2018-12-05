"""
Suite 2.

- Norm computations
- ?

"""

import sys
sys.path.append('/home/scratch/kkorovin')

from utils import *
from flatness import compute_local_entropy

import torch
import argparse
import matplotlib.pyplot as plt


def compute_norm(pt, p=2):
    norm_pow = 0.
    for layer in pt.keys():
        norm_pow += torch.sum(torch.abs(pt[layer]) ** p)
    return norm_pow.pow(1/float(p)).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_names', type=str, nargs="+", help='experiment names')
    parser.add_argument('--labels', type=str, nargs="+", help='experiment params')  # (TODO: make loader)
    args = parser.parse_args()

    for exp_name, label in zip(args.exp_names, args.labels):
        pts = load_history(exp_name)["trajectory"]

        l2s = []
        for pt in pts:
            # l1s.append(compute_norm(pt, p=1))
            l2s.append(compute_norm(pt, p=2))

        plt.plot(range(len(l2s)), l2s, label=label)
        plt.scatter(range(len(l2s)), l2s)
    plt.xlabel("Epoch group")
    plt.ylabel("L2 norm")
    plt.legend()
    plt.savefig("norm_trajectories")

    # Compute local entropy
    for exp_name, label in zip(args.exp_names, args.labels):
        entr = compute_local_entropy(exp_name, "lenet", "mnist",
                                     gamma=100, n_trials=10)
        print("Entropy: {:.4f}".format(entr))
