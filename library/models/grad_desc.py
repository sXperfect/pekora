import torch
from torch import optim
from torch.functional import F
import torchmetrics
import torchmetrics.functional as tmF
from .. import utils
from .base import _BaseModel

class GradientDescent(_BaseModel):
    def __init__(self,
        arch,
        lr:float=0.0,
        alpha=-0.25,
        loss='mse',
        loss_kwargs:dict=None,
        optimizer='adam',
        optimizer_kwargs:dict=None,
        scheduler=None,
        scheduler_kwargs:dict=None,
    ):

        super().__init__(
            arch,
            lr,
            alpha,
            loss,
            loss_kwargs=loss_kwargs,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )

    # def compute_D(self,
    #     row_ids,
    #     col_ids,
    # ):
    #     P_rows, P_cols = self.arch(row_ids, col_ids)
    #     pred_D = utils.edm_torch.compute_euc_dist_square(P_rows, P_cols)

    #     return pred_D

    # def compute_G(self,
    #     row_ids,
    #     col_ids,
    # ):
    #     P_rows, P_cols = self.arch(row_ids, col_ids)
    #     pred_G = utils.edm_torch.compute_gram(P_rows, P_cols)
        
    #     return pred_G

    def training_step(self, 
        batch, 
        batch_idx
    ):
        row_ids, col_ids, D_vals = self.arch.get_data(batch)

        D = self.compute_D(row_ids, col_ids)
        loss = self.loss_f(D, D_vals)

        self.log(
            f"main_loss",
            loss.item(),
            prog_bar=True,
        )

        return loss