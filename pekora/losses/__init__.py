from functools import partial
import torch
from torch.functional import F
from .spearman_approx import k_order_spearman_approx
from .reg_points import (
    MinimizeAdjecentPointDistance,
)
from .mds import (
    MultiDimensionalScaling,
    WeightedMultiDimensionalScaling,
)

SPEARMAN_WARN = "The use of Spearman Correlation as a loss function is not recommended due to runtime overhead!"

def select_loss(name, args=None, kwargs=None):
    if args is None:
        args = list()
    if kwargs is None:
        kwargs = dict()
    
    if name is None:
        f = None
    elif name == 'mse':
        f = partial(F.mse_loss, reduction='mean')
    elif name == 'sse':
        f = partial(F.mse_loss, reduction='sum')
    elif name == 'mae':
        f = partial(F.l1_loss, reduction='mean')
    elif name == 'sae':
        f = partial(F.l1_loss, reduction='sum')
    elif name in ['mds']:
        f = MultiDimensionalScaling(*args, **kwargs)
    
    return f

def select_reg(
    name, 
    args=None, 
    kwargs=None
):
    if args is None:
        args = list()
    if kwargs is None:
        kwargs = dict()
        
    if name is None:
        f = None
    elif name in ['ksa', 'k_order_spearman_approx']:
        f = partial(k_order_spearman_approx, *args, **kwargs)
    # elif name in ['MinimizeAbsDistanceAdjecentPoint']:
    #     f = MinimizeAbsDistanceAdjecentPoint(*args, **kwargs)
    elif name in ['MinimizeAdjecentPointDistance']:
        f = MinimizeAdjecentPointDistance(*args, **kwargs)
    # elif name in ['min_dist_var', "MinimizeDistanceVariationsAdjPoints"]:
    #     f = MinimizeDistanceVariationsAdjPoints(*args, **kwargs)
    else:
        raise ValueError(f"Invalid regularization function:{name}")
    
    return f
    