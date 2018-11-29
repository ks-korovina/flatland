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
import torch.nn as nn
from constants import DEVICE


############### Training History #################

def record_experiment(model_name, dataset_name, batch_size, lr, **kwargs):
    name = "model:{} data:{} bs:{} lr:{}".format(model_name, dataset_name, batch_size, int(-np.log10(lr)))
    enc_name = int(time())
    with open("log", "a") as f:
        f.write("Experiment {}: parameters {}\n".format(enc_name, name))
    return str(enc_name)


def init_history():
    return {"train_loss": [], "trajectory": []}


def update_history(data, history, check_name):
    """ Save history """
    history['train_loss'].append(data['train_loss'])
    # history['val_loss'].append(data['val_loss'])
    history['trajectory'].append(data['weights'])  # TODO: detach and convert to numpy array

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


############### Operations on points #################

def average_with_weights(thetas, weights):
    # check that number of params match
    n_models = len(thetas)
    assert len(thetas[0]) == len(thetas[1])

    res = {}
    layer_names = thetas[0].keys()
    for layer in layer_names:
        weighted_sum = torch.stack([weights[i] * thetas[i][layer] for i in range(n_models)], dim=0).sum(0)
        res[layer] = weighted_sum
    return res


def generate_simplex_combs(thetas, n_points=100):
    """
    Generator for points (parameter settings) with
        W_i = convex_comb(W_i^k for k=1,2,[3])
    thetas - list of two or three parameter settings
    """
    # assert len(thetas) == len(weights)
    assert len(thetas) in (2, 3)
    for _ in range(n_points):
        weights = np.random.random_sample(size=3)
        weights /= weights.sum()
        yield average_with_weights(thetas, weights), weights

# TODO: add other utilities here

def compute_loss(model, xs, ys):
    """ Use a loaded model """
    logits = model(xs)
    loss = nn.CrossEntropyLoss()(logits, ys)
    return loss.item()


def compute_approx_train_loss(model, data_loader, break_after=1):
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


