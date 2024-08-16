import numpy as np
import torch
from .consts import AVAIL_NP_F, AVAIL_TORCH_F
from ._base_binary_op import BaseBinaryOP

VALID_TERM = ['abs', 'square']
VALID_REDUCTION = ['mean', 'sum']

class MultiDimensionalScaling(BaseBinaryOP):

    def __init__(self,
        term='abs',
        reduction='mean'
    ):
        super().__init__()

        assert term in VALID_TERM, \
            f"Invalid term function:{term}"

        assert reduction in VALID_REDUCTION, \
            f"Invalid reduction function:{reduction}"

        self.term = term
        self.reduction = reduction

    def _call_numpy(self, input, target):
        term_f = AVAIL_NP_F[self.term]

        numerator = term_f(target-input)
        denominator = term_f(target)

        loss = numerator/denominator

        try:
            agg_f = AVAIL_NP_F[self.reduction]
        except:
            raise ValueError("Invalid reduction method for NumPy input")

        return agg_f(loss)

    def _call_torch(self, input, target):
        term_f = AVAIL_TORCH_F[self.term]

        numerator = term_f(target-input)
        denominator = term_f(target)

        loss = numerator/denominator

        try:
            agg_f = AVAIL_TORCH_F[self.reduction]
        except:
            raise ValueError("Invalid reduction method for PyTorch input")

        return agg_f(loss)

class WeightedMultiDimensionalScaling(BaseBinaryOP):
    #TODO: Check the operation!
    def __init__(self,
        reduction='mean',
        weight_exp=1.0
    ):

        super().__init__()

        self.weight_exp=1.0

    def __call__(self, input, target):
        loss = (target - input).square()
        weights = torch.pow(target, self.weight_exp)
        loss = torch.sum(weights * loss) / torch.sum(weights)
        return loss

    def _call_numpy(self, input, target):
        loss = np.abs(target - input) / target

        try:
            _f = AVAIL_NP_F[self.reduction]
        except:
            raise ValueError("Invalid reduction method for NumPy input")

        return _f(loss)

    def _call_torch(self, input, target):
        loss = (target - input).abs() / target.abs()

        try:
            _f = AVAIL_TORCH_F[self.reduction]
        except:
            raise ValueError("Invalid reduction method for PyTorch input")

        return _f(loss)