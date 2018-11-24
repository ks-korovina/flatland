"""
Main training script

@author: kkorovin@cs.cmu.edu

TODO:
* refactor for readability (migrate to utils etc)
* better checkpoint/history saving strategy?

"""

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

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", DEVICE)


def update_history(history, history_name):
    # save history
    history['loss'].append(avg_loss)
    # TODO: detach and convert to numpy array
    history['point'].append(deepcopy(model.parameters()))

    os.makedirs("./checkpoints/", exist_ok=True)
    with open("./checkpoints/"+ history_name + ".pkl", "wb") as f:
        pkl.dump(history, f)


def load_history(history, history_name):
    """
    Load history with this function;
    A checkpoint can be loaded into an initialized model with .load(name)
    
    """
    pass

# checkpoint can be loaded into an initialized model with .load(name)

def main():
    batch_size = 32
    num_epochs = 10
    val_period = 5
    save_data_period = 5
    clip = 5.

    check_name = "nov24"  # name of current checkpoint/run
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

    history = {"loss": [], "point": []}

    # model.load(check_name)

    count = 0
    for epoch in range(num_epochs):
        print("Starting training epoch {}".format(epoch+1))

        model.train()

        running_loss = 0.
        num_batches = len(train_loader)

        for (xs, ys) in tqdm(train_loader):
            xs, ys = xs.cuda(), ys.cuda()
            optimizer.zero_grad()
            logits = model(xs)

            loss = cross_ent(logits, ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            running_loss += loss.item()
            if np.isnan(running_loss):
                print("Loss is nan")
                exit(0)

        avg_loss = running_loss / num_batches
        scheduler.step(avg_loss)
        model.save(check_name)

        print("Epoch {} loss: {:.3f}".format(epoch+1, avg_loss))

        if epoch % save_data_period == 0:
            # save history
            history['loss'].append(avg_loss)
            # TODO: detach and convert to numpy array
            history['point'].append(deepcopy(model.parameters()))
            with open(history_name + ".pkl", "wb") as f:
                pkl.dump(history, f)

        if epoch % val_period == 0:
            # TODO: validation
            pass


    # after the end of all epochs, last checkpoint is saved


if __name__=="__main__":
    main()





