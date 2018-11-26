"""
Main training script

@author: kkorovin@cs.cmu.edu

TODO:
* refactor for readability (migrate to utils etc)
* better checkpoint/history saving strategy?
* make a log file with setting (model, dataset, bs, lr, ...) --> exp name mappings

"""
import os

import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import get_data_loader
from models import VGG
from utils import *

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", DEVICE)


# checkpoint can be loaded into an initialized model with .load(name)

def run_training(exp_name="nov24", batch_size=32, num_epochs=10,
                 val_period=5, save_hist_period=5):
    """
    For now only one model (vgg-16).
    """

    # name of current checkpoint/run
    check_name = exp_name
    history_name = check_name+"_history"

    # setup model, optimizer and logging
    model = VGG("VGG16").to(DEVICE)
    optimizer = Adam(params=model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, patience=3,
                                  threshold=0.1, min_lr=1e-5)

    cross_ent = nn.CrossEntropyLoss()

    # load data
    train_loader = get_data_loader("cifar10", "train", batch_size)
    val_loader   = get_data_loader("cifar10", "val", batch_size)

    history = init_history()
    update_history({"train_loss": 1e10, "weights": deepcopy(model.state_dict())},
                    history, history_name)

    # model.load_state_dict(model.state_dict())
    exit(0)

    # model.load(check_name)

    count = 0
    for epoch in range(num_epochs):
        model.train()
        print("Starting training epoch {}".format(epoch+1))
        
        running_loss = 0.
        num_batches = len(train_loader)

        for (xs, ys) in tqdm(train_loader):
            xs, ys = xs.to(DEVICE), ys.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xs)

            loss = cross_ent(logits, ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            running_loss += loss.item()
            if np.isnan(running_loss):
                print("Loss is nan")
                exit(0)

        avg_loss = running_loss / num_batches
        scheduler.step(avg_loss)
        model.save(check_name)

        print("Epoch {} loss: {:.3f}".format(epoch+1, avg_loss))

        if epoch % save_hist_period == 0:
            update_history({"train_loss": avg_loss,
                            "weights": deepcopy(model.parameters())},
                             history, history_name)

        if epoch % val_period == 0:
            model.eval()
            for (xs, ys) in tqdm(val_loader):
                xs, ys = xs.to(DEVICE), ys.to(DEVICE)
                logits = model(xs)
                # TODO: predictions

    # after the end of all epochs, last checkpoint has been saved

if __name__=="__main__":
    run_training()





