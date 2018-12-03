"""
Pytorch datasets for MNIST and CIFAR10
(other datasets can be added later, too).
User interface is provided with get_data_loader function.

Author: kkorovin@cs.cmu.edu

TODO
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10

from constants import DEVICE

# New Dataset classes can be created using this interface:
# class CIFAR10(Dataset):
#   def __init__(self):
#       pass
#   def __len__(self):
#       pass
#   def __getitem__(self):
#       pass

def replace_labels(dataset, num_classes):
    """ inplace modifies the dataset """
    for i in range(len(dataset.train_labels)):
        if np.random.random_sample() < 0.5:
            dataset.train_labels[i] = np.random.choice(num_classes)


def get_data_loader(dataset_name, mode, batch_size=100):
    dataset_params = dataset_name.split("_")
    if len(dataset_params) == 1:
        dataset_name, random_labels = dataset_params[0], False
    elif len(dataset_params) == 2:
        dataset_name, random_labels = dataset_params[0], True
    else:
        raise ValueError("Wrong dataset name format {}, \
                         should be datasetname or datasetname_random".format(dataset_name))

    if dataset_name == "mnist":
        if mode == "train":
            transform_train = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_data = MNIST(root='./data', train=True, download=True, transform=transform_train)
            if random_labels: replace_labels(train_data, 10)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
            return train_loader

        elif mode in ("val", "dev"):
            transform_test = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
            ])
            val_data = MNIST(root='./data', train=False, download=True, transform=transform_test)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
            return val_loader

        elif mode == "test":
            raise ValueError("No need to use a test dataset.")

    elif dataset_name == "cifar10":
        if mode == "train":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_data = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            if random_labels: replace_labels(train_data, 10)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
            return train_loader

        elif mode in ("val", "dev"):
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            val_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
            return val_loader

        elif mode == "test":
            raise ValueError("No need to use a test dataset.")

        else:
            ValueError("Unknown mode {}".format(mode))
    else:
        raise ValueError("Unknown dataset {}".format(dataset_name))

