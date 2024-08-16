import numpy as np
import torch
from torch.functional import F

AVAIL_TORCH_F = {
    'mean': torch.mean,
    'sum': torch.sum,
    'abs': torch.abs,
    'l1': F.l1_loss,
    'square': torch.square,
    'l2': F.mse_loss,
    'power': torch.pow,
    'log': torch.log,
}

AVAIL_NP_F = {
    'mean': np.mean,
    'sum': np.sum,
    'abs': np.abs,
    'l1': np.abs,
    'square': np.square,
    'l2': np.square,
    'power': np.power,
    'log': np.log
}