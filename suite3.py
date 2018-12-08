"""
Suite 3. Norm computations

"""

import sys
sys.path.append('/home/scratch/kkorovin')

from utils import *

import torch
import argparse
import matplotlib.pyplot as plt


def compute_norm(pt, p=2):
    norm_pow = 0.
    for layer in pt.keys():
        if is_float(pt[layer]):
            norm_pow += torch.sum(torch.abs(pt[layer]) ** p)
    return norm_pow.pow(1/float(p)).item()

def compute_grad_norm(pt, p=2):
    norm_pow = 0.
    for layer in pt.keys():
        if is_float(pt[layer]) and hasattr(pt[layer], "grad"):
            # issue: grads don't get saved in state_dict?
            norm_pow += torch.sum(torch.abs(pt[layer].grad) ** p)
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
    plt.savefig("norm_trajectories" + "_".join(s for s in args.exp_names))
    plt.clf()


    # for exp_name, label in zip(args.exp_names, args.labels):
    #     pts = load_history(exp_name)["trajectory"]

    #     l2s = []
    #     for pt in pts:
    #         # l1s.append(compute_norm(pt, p=1))
    #         l2s.append( np.log(compute_grad_norm(pt, p=2)) )

    #     plt.plot(range(len(l2s)), l2s, label=label)
    #     plt.scatter(range(len(l2s)), l2s)
    # plt.xlabel("Epoch group")
    # plt.ylabel("L2 norm of gradient, log")
    # plt.legend()
    # plt.savefig("grad_norm_trajectories" + "_".join(s for s in args.exp_names))

