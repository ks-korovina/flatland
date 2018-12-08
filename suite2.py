"""
Suite 2.

- Local entropy
- Hessian-based - TODO
- Reparametrization - TODO

"""

import sys
sys.path.append('/home/scratch/kkorovin')

from constants import DEVICE
from datasets import get_data_loader
from models import get_model
from utils import *
from flatness import compute_local_entropy, \
                     compute_c_epsilon_flatness, \
                     compute_spectral_sharpness

from flatness import _compute_local_entropy, _compute_c_epsilon_flatness
from flatness import compute_spectral_sharpness
from models import MnistMLP

import torch
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy


#==============================================================================
#           Reparametrization
#==============================================================================

def reparametrize_and_compute(dataset_name, break_after=-1):
    """
    Starting from a randomly initialized MLP,
    compute 1) approx radius of flatness before & after reparam
            2) local entropy before & after reparam
    """
    model = MnistMLP().to(DEVICE)
    network_params = model.state_dict()
    train_loader = get_data_loader(dataset_name, "train", 100)

    rad0 = np.mean(_compute_c_epsilon_flatness(model, train_loader, network_params,
                                               break_after=break_after))
    entr0 = _compute_local_entropy(model, train_loader, network_params,
                                   break_after=break_after)

    model = MnistMLP().to(DEVICE)
    network_params = model.state_dict()
    train_loader = get_data_loader(dataset_name, "train", 100)
    network_params["fc1.weight"] /= 5.
    network_params["fc2.weight"] *= 5.
    rad1 = np.mean(_compute_c_epsilon_flatness(model, train_loader, network_params,
                                               break_after=break_after))
    entr1 = _compute_local_entropy(model, train_loader, network_params,
                                   break_after=break_after)

    print("Radius of flatness before reparam: {:.3f}, after reparam: {:.3f}".format(rad0, rad1))
    print("Local entropy before reparam: {:.3f}, after reparam: {:.3f}".format(entr0, entr1))


def reparam_and_local_entropy(exp_name, model_name, dataset_name,
                              gamma=100, n_trials=10, break_after=-1):
    model = get_model(model_name).to(DEVICE)
    network_params = model.state_dict()
    network_params["features.0.weight"] /= 5.
    network_params["features.3.weight"] *= 5.
    train_loader = get_data_loader(dataset_name, "train", 100)
    entr = _compute_local_entropy(model, train_loader, network_params,
                                  break_after=break_after)
    return entr

def reparam_and_c_eps_flat(exp_name, model_name, dataset_name,
                           gamma=100, n_trials=10, break_after=-1):
    model = get_model(model_name).to(DEVICE)
    network_params = model.state_dict()
    network_params["features.0.weight"] /= 5.
    network_params["features.3.weight"] *= 5.
    train_loader = get_data_loader(dataset_name, "train", 100)
    rad = np.mean(_compute_c_epsilon_flatness(model, train_loader, network_params,
                                               break_after=break_after))
    return rad


#==============================================================================
#           Experiments
#==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_names', type=str, nargs="+", help='experiment names')
    parser.add_argument('--model', type=str, help='model name')  # (TODO: make loader)
    parser.add_argument('--dataset', type=str, help='dataset name')
    args = parser.parse_args()

    # Compute all measures of flatness for comparison
    entr, rad, eig = [], [], []
    for exp_name in args.exp_names:
        ############        Local entropy        #############
        entr.append( compute_local_entropy(exp_name, args.model, args.dataset,
                                     gamma=100, n_trials=10) )
        # entr = reparam_and_local_entropy(exp_name, args.model, args.dataset,
        #                              gamma=100, n_trials=10)

        ############            Radius           #############
        rad.append( compute_c_epsilon_flatness(exp_name, args.model, args.dataset,
                                         n_trials=10, break_after=10) )
        # rad = reparam_and_c_eps_flat(exp_name, args.model, args.dataset,
        #                              n_trials=10, break_after=10)

        ############          Spectral           #############
        eig.append( compute_spectral_sharpness(exp_name, args.model, args.dataset) )

    # Averaging over several re-runs
    print("Local entropy: {:.4f} +/- {:.4f}".format(np.mean(entr), np.std(entr)))
    print("Radius of flatness: {:.4f} +/- {:.4f}".format(np.mean(rad), np.std(rad)))
    print("Spectral sharpness: {:.4f} +/- {:.4f}".format(np.mean(eig), np.std(eig)))

    # Benefit of local entropy: reparametrization: TODO
    # reparametrize_and_compute("mnist", break_after=10)

