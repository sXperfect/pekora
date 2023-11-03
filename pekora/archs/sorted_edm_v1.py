import torch
from torch import nn
import numpy as np
from .edm import EDMArch

class SortedEDMArchV1(EDMArch):

    def __init__(self,
        row_ids:np.array,
        col_ids:np.array,
        C_vals:np.array,
        D_vals:np.array,
        shape,
        d:int=3,
        P=None,
        num_batches=1,
        precision=32,
        mean_free=True,
    ):
        
        super().__init__(
            row_ids,
            col_ids,
            C_vals,
            D_vals,
            shape,
            d=d,
            P=P,
            num_batches=num_batches,
            precision=precision,
            init_mean_free=mean_free,
        )
        
        self._sort_values()

    def _sort_values(self):
        sort_ids = np.argsort(self.D_vals)
        self.row_ids.data = self.row_ids.data[sort_ids]
        self.col_ids.data = self.col_ids.data[sort_ids]
        self.D_vals.data = self.D_vals.data[sort_ids]
        
        try:
            self.C_vals.data = self.C_vals.data[sort_ids]
        except:
            pass
        