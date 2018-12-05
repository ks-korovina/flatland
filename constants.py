"""
Unified imports

"""

import torch
N_WORKERS = 0
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", DEVICE)
