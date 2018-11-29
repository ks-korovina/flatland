""" Suite 1

Starting from a few checkpoints, compute flatness and and generalization.

@author: kkorovin@cs.cmu.edu

---------

TODO:
*

"""
from datasets import get_data_loader
from models import VGG
from utils import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", DEVICE)


def visualize_checkpoint_simplex(exp_names):
    """
    Given three points, plot surface over their convex combinations.
    """
    ps = []
    for exp_name in exp_names:
        last_trajectory_point = load_history(exp_name)['trajectory'][-1]
        ps.append(last_trajectory_point)

    model = VGG("VGG16")
    train_loader = get_data_loader("cifar10", "train", 100)

    x_simplex, y_simplex, losses = [], [], []
    for simplex_sample in generate_simplex_combs(ps, 10):
        network_params, simplex_point = simplex_sample
        model.load_params(network_params)
        loss = compute_approx_train_loss(model, train_loader)

        x_simplex.append(simplex_point[0])
        y_simplex.append(simplex_point[1])
        losses.append(loss)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(x_simplex, y_simplex, losses, label="loss surface interpolation")
    plt.savefig("trajectories")


def compute_c_epsilon_sharpness(exp_name, eps=1e-3):
    """
    Input: experiment name and parameter epsilon
    Returns: float sharpness
    """
    last_trajectory_point = load_history(exp_name)['trajectory'][-1]


def compute_c_epsilon_flatness(exp_name, eps=1e-3, n_trials=100):
    """
    Input: experiment name and parameter epsilon
    Returns: float sharpness
    """
    network_params = load_history(exp_name)['trajectory'][-1]
    for _ in range(n_trials):
        # sample a random direction in unit ball (how? every param group in unit ball?)
        # perturb network_params in this random direction
        # find max_eps_group for each group along perturbation (iterate with some step until increase is large)
        pass
    # take max of max_eps_group's?


if __name__ == "__main__":
    exp_names = ["1543460776", "1543460767", "1543460771"]
    visualize_checkpoint_simplex(exp_names)
