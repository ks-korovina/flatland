"""
Main training script

@author: kkorovin@cs.cmu.edu

TODO:
* refactor for readability (migrate to utils etc)
* better checkpoint/history saving strategy?
* make a log file with setting (model, dataset, bs, lr, ...) --> exp name mappings - done

"""

import sys
sys.path.append('/home/scratch/kkorovin')

import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from constants import DEVICE
from datasets import get_data_loader
from models import get_model
from utils import *

# checkpoint can be loaded into an initialized model with .load(name)

def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model to use')
    parser.add_argument('--dataset', type=str, help='dataset to use')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--n_epochs', default=15, type=int, help='number of training epochs')
    parser.add_argument('--verbose', default=False, type=bool, help='whether to display training statistics')
    args = parser.parse_args()
    return args


def run_training(model_name="vgg16",
                 dataset_name="cifar10",
                 batch_size=32, lr=1e-3, n_epochs=10,
                 save_hist_period=1, verbose=False):
    """
    For now only one model (vgg-16).

    Params:
    :model_name: "vgg{11,13,16,19}" or "lenet" (or "[...]_random")
    :dataset_name: "cifar10" or "mnist"
    :batch_size: int
    :lr: float
    :n_epochs: number of training epochs
    :save_hist_period: frequency with which points are saved

    """
    # name of current checkpoint/run
    check_name = record_experiment(model_name, dataset_name, batch_size, lr)

    # setup model, optimizer and logging
    model = get_model(model_name).to(DEVICE)
    optimizer = SGD(params=model.parameters(), lr=lr)
    # scheduler = ReduceLROnPlateau(optimizer, patience=3,
    #                               threshold=0.1, min_lr=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    cross_ent = nn.CrossEntropyLoss()

    # load data
    train_loader = get_data_loader(dataset_name, "train", batch_size)
    val_loader   = get_data_loader(dataset_name, "val", batch_size)

    history = init_history()

    update_history({"train_loss": float("inf"),
                    "val_acc": 0.,
                    "weights": deepcopy(model.state_dict())},
                     history, check_name)

    for epoch in range(n_epochs):
        model.train()
        if verbose: print("Starting training epoch {}".format(epoch+1))

        running_loss = 0.
        num_batches = len(train_loader)

        for (xs, ys) in train_loader:
            xs, ys = xs.to(DEVICE), ys.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xs)

            loss = cross_ent(logits, ys)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            running_loss += loss.item()
            if np.isnan(running_loss):
                print("Loss is nan")
                exit(0)

        avg_loss = running_loss / num_batches
        scheduler.step(avg_loss)
        model.save(check_name)

        if verbose: print("Epoch {} loss: {:.3f}".format(epoch+1, avg_loss))

        if epoch % save_hist_period == 0:
            model.eval()
            accs = []
            for (xs, ys) in val_loader:
                xs, ys = xs.to(DEVICE), ys.to(DEVICE)
                logits = model(xs)
                y_pred = logits.argmax(dim=1)
                batch_acc = (y_pred == ys).float().mean().item()
                accs.append(batch_acc)

            if verbose: print("Validation accuracy: {:.3f}".format(np.mean(accs)))
            update_history({"train_loss": avg_loss,
                            "val_acc": np.mean(accs),
                            "weights": deepcopy(model.state_dict())},
                             history, check_name)

    print("Last avg loss {}, eval acc {}".format(avg_loss, np.mean(accs)))
    # after the end of all epochs, last checkpoint has been saved


if __name__=="__main__":
    args = parse_train_args()
    run_training(args.model, args.dataset,
                 batch_size=args.batch_size, lr=args.lr,
                 n_epochs=args.n_epochs, verbose=args.verbose)

