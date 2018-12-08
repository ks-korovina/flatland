"""
Experiment Suite 1 functions.
Starting from a few checkpoints, compute flatness and and generalization.

@author: kkorovin@cs.cmu.edu

TODO:
* compute flatness/sharpness measures
* think of how to refactor flatness measure computations
* use meshes instead of triangulations (also needs a simplex grid) - done

"""
import sys
sys.path.append('/home/scratch/kkorovin')

from constants import DEVICE
from datasets import get_data_loader
from models import get_model
from utils import *
from flatness import compute_c_epsilon_flatness

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import matplotlib
matplotlib.rcParams.update({'font.serif': 'Times New Roman',
                            'font.size': 10.0,
                            'axes.labelsize': 'Medium',
                            'axes.labelweight': 'normal',
                            'axes.linewidth': 0.8,
                            'figure.figsize':[20,15]
                            })

def visualize_checkpoint_simplex(exp_names, model_name, dataset_name,
                                 cutoff=3.5, mode="grid", break_after=-1):
    """
    Given three points, plot surface over their convex combinations.
    """
    ps = []
    for exp_name in exp_names:
        last_trajectory_point = load_history(exp_name)['trajectory'][-1]
        ps.append(last_trajectory_point)

    model = get_model(model_name).to(DEVICE)
    train_loader = get_data_loader(dataset_name, "train", 100)

    # TODO: use meshgrid instead
    if mode == "triangle":
        x_simplex, y_simplex, losses = [], [], []
        for simplex_sample in generate_simplex_combs(ps, 3):
            network_params, simplex_point = simplex_sample
            model.load_params(network_params)
            loss = compute_approx_train_loss(model, train_loader)

            x_simplex.append(simplex_point[0])
            y_simplex.append(simplex_point[1])
            losses.append(loss)

        tri = mtri.Triangulation(x_simplex, y_simplex)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x_simplex, y_simplex, losses, triangles=tri.triangles, 
                        cmap=plt.cm.Spectral, label="loss surface interpolation")
        plt.savefig("surf_" + "_".join(s for s in exp_names))

    else:
        x = np.linspace(-0.4, 1.3, 50)
        y = np.linspace(-0.4, 1.3, 50)

        X, Y = np.meshgrid(x, y)
        Z = 1 - X - Y
        grid = simplex_grid(3, 25) / 25
        grid_val = []

        Z_ = []
        for i in tqdm(range(X.shape[0])):
            Z_ += [[]]
            for j in range(Y.shape[0]):
                weights = [X[i, j], Y[i, j], Z[i, j]]
                network_params = average_with_weights(ps, weights)
                model.load_params(network_params)
                loss = compute_approx_train_loss(model, train_loader,
                                                 break_after)
                Z_[i].append(loss)

        losses = np.array(Z_)

        # backup everything everything
        np.save("./data/X_" + "_".join(s for s in exp_names), X)
        np.save("./data/Y_" + "_".join(s for s in exp_names), Y)
        np.save("./data/Z_" + "_".join(s for s in exp_names), Z_)

        losses[losses > cutoff] = cutoff

        fig = plt.figure()
        cmap = matplotlib.cm.coolwarm
        cmap.set_bad('white', 1.)
        cmap.set_over('white', alpha=.1)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, losses, vmax=cutoff, rstride=1, cstride=1,
                        cmap=cmap, edgecolor='none', antialiased=True)
        ax.view_init(50, 200)  #225

        plt.savefig("surf_" + "_".join(s for s in exp_names))
        plt.clf()


def visualize_interpolated_trajectory(exp_names, model_name, dataset_name, break_after=-1):
    """ repeat for three runs """
    for exp_name in exp_names:
        trajectory = load_history(exp_name)['trajectory']

        model = get_model(model_name).to(DEVICE)
        train_loader = get_data_loader(dataset_name, "train", 100)

        res = []
        epochs = []
        for i in range(len(trajectory)-1):
            ps = [trajectory[i], trajectory[i+1]]
            for alph in [0., 0.2, 0.4, 0.6, 0.8]:
                weights = [alph, 1-alph]
                network_params = average_with_weights(ps, weights)
                model.load_params(network_params)
                loss = compute_approx_train_loss(model, train_loader,
                                                 break_after)
                epochs.append(i+1+alph)
                res.append(loss)

        plt.plot(epochs, res)

    plt.xlabel("Interpolated epoch")
    plt.ylabel("Approx loss")
    plt.savefig("loss_interpolation_" + exp_name)
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_names', type=str, nargs=3, help='experiment names')
    parser.add_argument('--model', default="lenet", type=str, help='model to use')
    parser.add_argument('--dataset', default="mnist", type=str, help='dataset to use')
    args = parser.parse_args()

    visualize_checkpoint_simplex(args.exp_names, args.model, args.dataset,
                                 break_after=20)
    for exp_name in args.exp_names:
        compute_c_epsilon_flatness(exp_name, args.model, args.dataset,
                                   n_trials=10, break_after=10)
        break

    # Average trajectories
    visualize_interpolated_trajectory(args.exp_names, args.model, args.dataset,
                                      break_after=10)


