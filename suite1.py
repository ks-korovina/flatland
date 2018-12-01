"""
Experiment Suite 1 functions.
Starting from a few checkpoints, compute flatness and and generalization.

@author: kkorovin@cs.cmu.edu

TODO:
* compute flatness/sharpness measures
* think of how to refactor flatness measure computations
* use meshes instead of triangulations (also needs a simplex grid)

"""

from constants import DEVICE
from datasets import get_data_loader
from models import get_model
from utils import *
from flatness import compute_c_epsilon_flatness

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri


def visualize_checkpoint_simplex(exp_names, model_name, dataset_name):
    """
    Given three points, plot surface over their convex combinations.
    """
    ps = []
    for exp_name in exp_names:
        last_trajectory_point = load_history(exp_name)['trajectory'][-1]
        ps.append(last_trajectory_point)

    model = get_model(model_name)
    train_loader = get_data_loader(dataset_name, "train", 100)

    # TODO: use meshgrid instead
    x_simplex, y_simplex, losses = [], [], []
    for simplex_sample in generate_simplex_combs(ps, 1000):
        network_params, simplex_point = simplex_sample
        model.load_params(network_params)
        loss = compute_approx_train_loss(model, train_loader)

        x_simplex.append(simplex_point[0])
        y_simplex.append(simplex_point[1])
        losses.append(loss)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    tri = mtri.Triangulation(x_simplex, y_simplex)
    ax.plot_trisurf(x_simplex, y_simplex, losses, triangles=tri.triangles, 
                    cmap=plt.cm.Spectral, label="loss surface interpolation")
    plt.savefig("Surface of exps " + str(exp_names))


if __name__ == "__main__":
    # # dummy
    # exp_names = ["1543460776", "1543460767", "1543460771"]
    # visualize_checkpoint_simplex(exp_names)

    # 1-epoch mnist on lenet
    exp_names = ["1543467966", "1543468095", "1543468159"]
    # visualize_checkpoint_simplex(exp_names, "lenet", "mnist")
    for exp_name in exp_names:
        compute_c_epsilon_flatness(exp_name, "lenet", "mnist", n_trials=10)
