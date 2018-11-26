"""
Helper functions for training and running experiments

TODO:
* add validation loss tracking
* add plotting utils

"""

import os
import torch
import numpy as np
import pickle as pkl


############### Training History #################

def init_history():
    return {"train_loss": [], "trajectory": []}


def update_history(data, history, history_name):
    """ Save history """

    history['train_loss'].append(data['train_loss'])
    # history['val_loss'].append(data['val_loss'])
    history['trajectory'].append(data['weights'])  # TODO: detach and convert to numpy array

    os.makedirs("./checkpoints/", exist_ok=True)
    with open("./checkpoints/"+ history_name + ".pkl", "wb") as f:
        pkl.dump(history, f)


def load_history(history_name):
    """
    Load history with this function;
    A checkpoint can be loaded into an initialized model with .load(name)
    
    """
    with open("./checkpoints/"+ history_name + ".pkl", "rb") as f:
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
    assert len(thetas) == len(weights)
    assert len(thetas) in (2, 3)
    for _ in range(n_points):
        weights = np.random.random_sample(size=3)
        weights /= weights.sum()
        yield average_with_weights(thetas, weights)

# TODO: add other utilities here

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


