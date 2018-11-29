"""
Unified imports

"""

import torch
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", DEVICE)