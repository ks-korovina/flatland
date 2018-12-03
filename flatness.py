"""

Flatness computations

@authors: kkorovin@cs.cmu.edu,
          Riccardo Fogliato

"""

import torch
import numpy as np
import numpy as np
import random
import time
import math

from constants import DEVICE
from datasets import get_data_loader
from models import get_model
from utils import *


def compute_c_epsilon_sharpness(exp_name, eps=1e-3):
    """
    Input: experiment name and parameter epsilon
    Returns: float sharpness
    """
    last_trajectory_point = load_history(exp_name)['trajectory'][-1]


def compute_c_epsilon_flatness(exp_name, model_name, dataset_name,
							   eps=1e-2, n_trials=100):
    """
    Input: experiment name and parameter epsilon
    Returns: float flatness
    """
    model = get_model(model_name).to(DEVICE)
    train_loader = get_data_loader(dataset_name, "train", 100)

    network_params = load_history(exp_name)['trajectory'][-1]
    model.load_params(network_params)
    loss_init = compute_approx_train_loss(model, train_loader)

    steps_to_border = []
    for _ in range(n_trials):
        # generate a unit vector of the shape the same as network_weights
        delta_params = random_like(network_params)
        loss_perturbed = loss_init
        step = 0
        while loss_perturbed - loss_init < eps * loss_init:
        	step += 1
        	perturbed_params = add_weights(network_params, delta_params, step)
        	model.load_params(perturbed_params)
        	loss_perturbed = compute_approx_train_loss(model, train_loader)  # <---- change break_after parameter here

        # this trial, we went on average 'steps' unit vectors away from starting point
        steps_to_border.append(step)

    # Display results:
    print("Approximate radius of flatness: {:.3f}".format(np.mean(steps_to_border)))
    return np.mean(steps_to_border)


def compute_local_entropy(exp_name, model_name, dataset_name,
                               gamma, n_trials=100):
    model = get_model(model_name).to(DEVICE)
    train_loader = get_data_loader(dataset_name, "train", 100)
    return monte_carlo(model, train_loader, gamma, n_trials)


def monte_carlo(model, data, network_params, gamma, n_trials):
    d = np.sum([v.numel() for (k,v) in network_params.items()])

    avg = 0.
    for i in range(n_trials):
        sample = gaussian_like(network_params, 1/gamma)
        loss = compute_approx_train_loss(model, data)
        avg += np.exp(-loss) / num_trials

    Z = d/2 * np.log(2 * math.pi) + 0.5 * np.log(d/gamma)
    loss = Z * np.log(avg)

    return loss


if __name__=="__main__":
    gamma = 0.2
    n = 11 # number of samples
    d = 1000000
    w = np.arange(d)

    start = time.time()
    ret = monte_carlo(n, w, gamma)
    print(ret)
    end = time.time()
    print(np.round(end-start,2))


