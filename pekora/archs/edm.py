import torch
from torch import nn
import numpy as np

class EDMArch(torch.nn.Module):

    def __init__(self,
        row_ids:np.array,
        col_ids:np.array,
        C_vals:np.array,
        D_vals:np.array,
        shape,
        d:int=3,
        P=None,
        P_min=-1,
        P_max=1,
        num_batches=1,
        precision=64,
        init_mean_free=True,
    ):

        super().__init__()
        self.shape = shape
        self.d = d
        self.init_mean_free = init_mean_free

        self.init_P(P, P_min, P_max)

        self.num_batches = num_batches
        
        #? row_ids is nn.Parameter for ".to(device)"
        self.register_buffer(
            "row_ids", 
            torch.from_numpy(row_ids).long()
        )

        #? col_ids is nn.Parameter for ".to(device)"
        self.register_buffer(
            "col_ids", 
            torch.from_numpy(col_ids).long()
        )


        #? D_vals is nn.Parameter for ".to(device)"
        self.register_buffer(
            "D_vals", 
            torch.from_numpy(D_vals)
        )
        if precision != 64:
            self.D_vals = self.D_vals.float()
        
        try:
            self.register_buffer(
                "C_vals", 
                torch.from_numpy(C_vals)
            )
            if precision != 64:
                self.C_vals = self.C_vals.float()
        except:
            self.C_vals = None
            
        unique_ids = np.unique(np.concatenate([row_ids, col_ids]))
        mask = np.zeros(self.n, dtype=bool)
        mask[unique_ids] = 1
        # self.mask = mask
        self.register_buffer(
            "mask", 
            torch.from_numpy(mask)
        )

    @property
    def n(self):
        return np.amax(self.shape)

    def init_P(self, P, P_min, P_max):
        if P is None:
            P = torch.Tensor(self.n, self.d).uniform_(P_min, P_max)

        self.P = nn.Parameter(P)

    def __len__(self):
        return len(self.D_vals)

    def shuffle(self):
        pass

    def get_data(self, idx, with_C=False):
        data_per_batch = np.ceil(len(self)/self.num_batches).astype(np.int32)
        start_idx = data_per_batch*(idx+0)
        end_idx = data_per_batch*(idx+1)

        row_ids = self.row_ids[start_idx:end_idx]
        col_ids = self.col_ids[start_idx:end_idx]
        D_vals = self.D_vals[start_idx:end_idx]
        
        if with_C:
            try:
                C_vals = self.C_vals[start_idx:end_idx]
            except:
                raise ValueError("C_vals is None!")
            
            return row_ids, col_ids, C_vals, D_vals
        else:
            return row_ids, col_ids, D_vals
        
    def get_all_data(self):
        return (
            self.row_ids,
            self.col_ids,
            self.D_vals
        )

    def get_points(self,
        ids,
        detach=False
    ):
        if detach:
            return self.P.detach()[ids, :]
        else:
            return self.P[ids, :]

    def forward(self,
        row_ids,
        col_ids
    ):
        return self.get_points(row_ids), self.get_points(col_ids)