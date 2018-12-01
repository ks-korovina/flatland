"""

Flatness computations

@author: kkorovin@cs.cmu.edu

"""

import torch
import numpy as np

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
							   eps=1e-3, n_trials=100):
    """
    Input: experiment name and parameter epsilon
    Returns: float flatness
    """
    model = get_model(model_name)
    train_loader = get_data_loader(dataset_name, "train", 100)

    network_params = load_history(exp_name)['trajectory'][-1]
    model.load_params(network_params)
    loss_init = compute_approx_train_loss(model, train_loader, 1)

    steps_to_border = []
    for _ in range(n_trials):
        # generate a unit vector of the shape the same as network_weights
        delta_params = random_like(network_params)
        loss_perturbed = loss_init
        step = 0
        while loss_perturbed - loss_init < eps:
        	step += 1
        	perturbed_params = add_weights(network_params, delta_params, step)
        	model.load_params(perturbed_params)
        	loss_perturbed = compute_approx_train_loss(model, train_loader, 1)  # <---- change break_after parameter here

        # this trial, we went on average 'steps' unit vectors away from starting point
        steps_to_border.append(step)

    # Display results:
    print("Approximate radius of flatness: {:.3f}".format(np.mean(steps_to_border)))
    return np.mean(steps_to_border)

