"""
Helper functions for training and running experiments.

@author: kkorovin@cs.cmu.edu

TODO:
* add validation loss tracking
* add plotting utils

"""

import os
import torch
from time import time
import numpy as np
import pickle as pkl
import math
from scipy.special import comb
import numpy as np

import torch.nn as nn
from constants import DEVICE


############### Training History #################

def record_experiment(model_name, dataset_name, batch_size, lr, **kwargs):
    name = "model:{} data:{} bs:{} lr:10^-{}".format(model_name, dataset_name, batch_size, int(-np.log10(lr)))
    enc_name = int(time())
    exp_description = "Experiment {}: parameters {}".format(enc_name, name)
    with open("log", "a") as f:
        f.write(exp_description + "\n")
    print(exp_description)
    return str(enc_name)


def init_history():
    return {"train_loss": [], "val_acc": [], "trajectory": []}


def update_history(data, history, check_name):
    """ Save history """
    history['train_loss'].append(data['train_loss'])
    history['val_acc'].append(data['val_acc'])
    history['trajectory'].append(data['weights'])

    os.makedirs("./checkpoints/", exist_ok=True)
    with open("./checkpoints/"+ check_name + ".pkl", "wb") as f:
        pkl.dump(history, f)


def load_history(check_name):
    """
    Load history with this function;
    A checkpoint can be loaded into an initialized model with .load(name)
    """
    with open("./checkpoints/"+ check_name + ".pkl", "rb") as f:
        history = pkl.load(f)
    return history


############### Operations on net parameters (weights) #################

def average_with_weights(thetas, weights):
    # check that number of params match
    n_models = len(thetas)
    assert len(thetas[0]) == len(thetas[1])

    res = {}
    layer_names = thetas[0].keys()
    for layer in layer_names:
        weighted_sum = torch.stack([weights[i] * thetas[i][layer]
                                    for i in range(n_models)], dim=0
                                   ).sum(0)
        res[layer] = weighted_sum
    return res


def add_weights(dw1, dw2, step=1.):
    res = {}
    for layer in dw1.keys():
        res[layer] = dw1[layer] + float(step) * dw2[layer]
    return res


def random_like(weight_dict):
    delta_weights = {}
    delta_norm = 0.
    for layer in weight_dict.keys():
        delta_layer = torch.randn(weight_dict[layer].size())
        delta_norm += torch.sum(delta_layer ** 2).item()
        delta_weights[layer] = delta_layer
    delta_norm = np.sqrt(delta_norm)
    return {k: (v/delta_norm).to(DEVICE) for k, v in delta_weights.items()}


def gaussian_like(weight_dict, cov):
    rand_weights = {}
    for layer in weight_dict.keys():
        rand_weights[layer] = torch.randn(weight_dict[layer].size()) * cov
    return rand_weights


def generate_simplex_combs(thetas, n_points=100):
    """
    Generator for points (parameter settings) with
        W_i = convex_comb(W_i^k for k=1,2,[3])
    thetas - list of two or three parameter settings
    """
    assert len(thetas) in (2, 3)
    for _ in range(n_points):
        weights = np.random.uniform(-0.1, 1.1, size=3)
        while weights.sum() > 1.2:
            weights = np.random.uniform(-0.1, 1.1, size=3)
        yield average_with_weights(thetas, weights), weights

# def simplex_grid(size=0.1):
#     grid_list = []
#     alpha_list = []
#     for a1 in np.arange(0,1,size):
#         max_rest = 1-a1
#         for a2 in np.arange(0,max_rest,size):
#             a3 = 1-a1-a2
#             coord = [sum(x) for x in zip([a2*2,0], [a3,a3*math.sqrt(3)])]
#             grid_list.append(coord)
#             alpha_list.append([a1,a2,a3])
#     return grid_list, alpha_list

def simplex_grid(m, n):
    L = comb(n+m-1, m-1, exact=True)
    out = np.empty((L, m), dtype=int)

    x = np.zeros(m, dtype=int)
    x[m-1] = n

    for j in range(m):
        out[0, j] = x[j]

    h = m

    for i in range(1, L):
        h -= 1

        val = x[h]
        x[h] = 0
        x[m-1] = val - 1
        x[h-1] += 1

        for j in range(m):
            out[i, j] = x[j]

        if val != 1:
            h = m

    return out


############### Loss computations #################

def compute_loss(model, xs, ys):
    """ Use a loaded model """
    logits = model(xs)
    loss = nn.CrossEntropyLoss()(logits, ys)
    return loss.item()


def compute_approx_train_loss(model, data_loader, break_after=-1):
    """
    Params:
    :break_after: how many batches to use for loss estimation; -1 means use all
    """
    running_loss = 0.
    n_batches = 0
    for (xs, ys) in data_loader:
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        running_loss += compute_loss(model, xs, ys)
        n_batches += 1

        if break_after > -1 and n_batches >= break_after:
            break

    loss = running_loss / n_batches
    return loss

############### Tests for operations #################

from models import VGG

def run_tests():
    h = load_history("nov24_history")
    trajectory = h['trajectory']
    thetas = [trajectory[0], trajectory[0]]
    res = average_with_weights(thetas, [0.5, 0.5])
    # print(res)

    model = VGG("VGG16")
    model.load_params(res)


############### Run tests #################

if __name__=="__main__":
    run_tests()

