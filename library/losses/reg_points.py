import numpy as np
import torch
from torch.functional import F
from .. import utils

def _log_var(D):
    return torch.var(torch.log(D))

def _log_std(D):
    return torch.std(torch.log(D))

_AVAIL_AGGR_F = {
    'std': torch.std,
    'var': torch.var,
    'log_std': _log_std,
    'log_var': _log_var
}

class GeometricCenteringPoints(object):
    def __init__(self,
        op='square'             
    ):
            
        if op == 'square':
            self._f = torch.square
        elif op == 'abs':
            self._f = torch.abs
        else:
            f"Valid op is either 'square' or 'abs'!: {op}"
    
    def __call__(self, P):
        
        centroid_P = P.mean(axis=0)
        normed_centroid_P = self._f(centroid_P).sum()
        
        return normed_centroid_P

class MinimizeAdjecentPointDistance:
    def __init__(self,
        mode='square',
        unique_point_ids=None,
    ):
        
        if mode == 'square':
            self._f = torch.square
        elif mode == 'abs':
            self._f = torch.abs
        else:
            raise ValueError(f'Invalid mode:{mode}')
        
        if unique_point_ids is not None:
            mask = np.zeros(unique_point_ids.max()+1, dtype=bool)
            mask[unique_point_ids] = 1
            mask = np.logical_and(
                mask[:-1], 
                mask[1:]
            )
            self.mask = torch.tensor(mask)
        else:
            self.mask = None
            
    def __call__(self, 
        P,
        mask=None
    ):
        P_prev = P[:-1, :]
        P_next = P[1:, :]
        
        assert not (mask is not None and self.mask is not None), \
            "Both mask and self.mask exist! Please disable one of it"
        
        if self.mask is not None:
            P_prev = P_prev[self.mask]
            P_next = P_next[self.mask]
        
        elif mask is not None:
            valid_P_mask = torch.bitwise_and(mask[:-1], mask[1:])
        
            P_prev = P_prev[valid_P_mask]
            P_next = P_next[valid_P_mask]
            
        P_diff = P_prev-P_next
        P_diff = self._f(P_diff)
        
        loss = P_diff.sum(axis=1).mean()
        
        return loss
    