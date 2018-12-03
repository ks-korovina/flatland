"""
Suite 2.

- Norm computations
- ?

"""

import sys
sys.path.append('/home/scratch/kkorovin')

from utils import *
import torch

def compute_norm(pt, p=2):
	norm_pow = 0.
	for layer in pt.keys():
		norm_pow += torch.sum(torch.abs(pt[layer]) ** p)
	return norm_pow.pow(1/float(p)).item()

pts = load_history("1543633716")["trajectory"]

for pt in pts:
	print(compute_norm(pt, p=2))
	print(compute_norm(pt, p=1))
