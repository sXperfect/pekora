from os.path import join, exists
import json
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class SparseMatrixDataset(Dataset, ABC):
    def __init__(self, 
        row_ids:np.array,
        col_ids:np.array,
        C_vals:np.array,
        D_vals:np.array,
        shape,
        ret_C_vals:bool=False,
    ):
        
        super(SparseMatrixDataset,self).__init__()
        
        assert len(row_ids) == len(col_ids) == \
               len(C_vals) == len(D_vals), \
                   "Invalid array length!"
        
        self.row_ids = row_ids
        self.col_ids = col_ids
        self.C_vals = C_vals
        self.D_vals = D_vals
        self.shape = shape
        self.ret_C_vals = ret_C_vals
        
    def __len__(self):
        return len(self.row_ids)
    
    @property
    def greatest_dim(self):
        return np.amax(self.shape)

    def __getitem__(self, idx):
        row_id = self.row_ids[idx]
        col_id = self.col_ids[idx]
        D_val = self.D_vals[idx]
        
        if self.ret_C_vals:
            C_val = self.C_vals[idx]
            return row_id, col_id, C_val, D_val
        else:
            return row_id, col_id, D_val