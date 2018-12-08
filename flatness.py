"""

Flatness computations

@authors: kkorovin@cs.cmu.edu,
          Riccardo Fogliato

- Radius of flatness measures
- Local entropy
- Hessian-based (spectral norm of Hessian):
  credits go to the following open-source project:
  https://github.com/amirgholami/HessianFlow
- TODO

"""

import torch
import torch.nn as nn
import numpy as np
import random
import time
import math

from constants import DEVICE
from datasets import get_data_loader
from models import get_model
from utils import *


#==============================================================================
#           Radii of flatness
#==============================================================================

def compute_c_epsilon_sharpness(exp_name, eps=1e-3):
    """
    Input: experiment name and parameter epsilon
    Returns: float sharpness
    """
    raise NotImplementedError("Use C-epsilon flatness instead.")


def compute_c_epsilon_flatness(exp_name, model_name, dataset_name,
                               eps=0.05, n_trials=100, break_after=-1):
    """
    Input: experiment name and parameter epsilon
    Returns: float flatness
    """
    model = get_model(model_name).to(DEVICE)
    # bs here is chosen by processing speed:
    train_loader = get_data_loader(dataset_name, "train", 100)
    network_params = load_history(exp_name)['trajectory'][-1]

    # call helper
    steps_to_border = _compute_c_epsilon_flatness(model, train_loader, network_params,
                                                  eps, n_trials, break_after)

    # Display results:
    print("Approximate radius of flatness: {:.3f} +/- {:.3f}".format(np.mean(steps_to_border),
                                                                     np.std(steps_to_border)))
    return np.mean(steps_to_border)


def _compute_c_epsilon_flatness(model, data, network_params,
                                eps=0.05, n_trials=100, break_after=-1):
    model.load_params(network_params)
    loss_init = compute_approx_train_loss(model, data, break_after)

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
            loss_perturbed = compute_approx_train_loss(model, data, break_after)

        # this trial, we went on average 'steps' unit vectors away from starting point
        steps_to_border.append(step)

    return steps_to_border


#==============================================================================
#           Local entropy
#==============================================================================

def compute_local_entropy(exp_name, model_name, dataset_name,
                          gamma=100., n_trials=100, break_after=-1):
    model = get_model(model_name).to(DEVICE)
    network_params = load_history(exp_name)['trajectory'][-1]
    train_loader = get_data_loader(dataset_name, "train", 100)
    return _compute_local_entropy(model, train_loader, network_params, gamma, n_trials)


def _compute_local_entropy(model, data, network_params,
                           gamma=100., n_trials=100, break_after=-1):
    model.load_params(network_params)
    d = np.sum([v.numel() for (k,v) in network_params.items()])
    print("Number of parameters in the network:", d)

    avg = 0.
    for i in range(n_trials):
        perturbed_params = gaussian_like(network_params, 1/gamma)
        model.load_params(perturbed_params)
        loss = compute_approx_train_loss(model, data, break_after)
        avg += np.exp(-loss) / n_trials

    # Z = d/2 * np.log(2 * math.pi) + 0.5 * np.log(d/gamma)
    # loss = Z + 
    loss = np.log(avg)

    return loss


#==============================================================================
#           Second-order
#==============================================================================

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i,p in enumerate(params):
        params[i].data.add_(update[i] * alpha) 
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v,v)
    s = s ** 0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v

def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        params.append(param)
        if param.grad is None:
            continue
        grads.append(param.grad + 0.)
    return params, grads

def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=True)
    return hv

def get_eigen(model, inputs, targets, criterion, maxIter=50, tol=1e-3):
    """
    Compute the top eigenvalues of model parameters and 
    the corresponding eigenvectors.
    """
    model.eval()  # don't forget to return to train() afterwards

    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward(create_graph=True)

    params, gradsH = get_params_grad(model)
    v = [torch.randn(p.size()).to(DEVICE) for p in params]
    v = normalization(v)

    eigenvalue = None

    for i in range(maxIter):
        model.zero_grad()
        Hv = hessian_vector_product(gradsH, params, v)
        eigenvalue_tmp = group_product(Hv, v).cpu().item()
        v = normalization(Hv)
        if eigenvalue is None:
            eigenvalue = eigenvalue_tmp
        else:
            if abs(eigenvalue-eigenvalue_tmp) < tol:
                return eigenvalue_tmp, v
            else:
                eigenvalue = eigenvalue_tmp
    return eigenvalue, v


def compute_spectral_sharpness(exp_name, model_name, dataset_name):
    """ Compute average top eigenvalue over a few batches """
    model = get_model(model_name).to(DEVICE)
    network_params = load_history(exp_name)['trajectory'][-1]
    model.load_params(network_params)
    train_loader = get_data_loader(dataset_name, "train", 100)

    first = True
    for data in train_loader:
        if first:
            inputs, targets = data
            first = False
    # criterion = compute_loss(model, inputs, targets)
    eigenvalue, eigenvector = get_eigen(model, inputs, targets,
                                        nn.CrossEntropyLoss(), maxIter=10, tol=1e-2)
    return eigenvalue


if __name__=="__main__":
    pass

