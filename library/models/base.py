import abc
import torch
from pytorch_lightning import LightningModule
from ..losses import select_loss
from ..optimizers import select_optimizer
from ..schedulers import select_scheduler
from .. import utils

class _BaseModel(abc.ABC, LightningModule):
    def __init__(
        self,
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
        super().__init__()

        self.alpha = alpha
        self.arch = arch
        self.lr = lr

        self.optimizer_name = optimizer
        if optimizer_kwargs is not None:
            self.optimizer_kwargs = optimizer_kwargs
        else:
            self.optimizer_kwargs = {}

        self.scheduler_name = scheduler
        if scheduler_kwargs is not None:
            self.scheduler_kwargs = scheduler_kwargs
        else:
            self.scheduler_kwargs = {}

        self.loss_name = loss
        self.loss_f = select_loss(
            self.loss_name, 
            kwargs=loss_kwargs
        )

    def configure_optimizers(self):
        optimizer = select_optimizer(
            self.optimizer_name,
            self.parameters(),
            lr=self.lr,
            kwargs=self.optimizer_kwargs
        )

        lr_scheduler = select_scheduler(
            self.scheduler_name,
            optimizer,
            kwargs=self.scheduler_kwargs
        )

        if lr_scheduler is None:
            return optimizer
        else:
            lr_scheduler = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [lr_scheduler]

    def compute_G(self,
        row_ids,
        col_ids,
    ):
        P_rows, P_cols = self.arch(row_ids, col_ids)
        pred_G = utils.edm_torch.compute_gram(P_rows, P_cols)

        return pred_G

    def compute_G_nograd(self,
        row_ids,
        col_ids,
    ):
        with torch.no_grad():
            pred_G = self.compute_G(
                row_ids,
                col_ids
            )

        return pred_G

    def compute_D(self,
        row_ids,
        col_ids,
    ):
        P_rows, P_cols = self.arch(row_ids, col_ids)
        pred_D = utils.edm_torch.compute_euc_dist_square(P_rows, P_cols)

        return pred_D
    
    @torch.no_grad()
    def compute_D_nograd(self,
        row_ids,
        col_ids,
    ):
        P_const, P_var = self.arch(row_ids, col_ids)
        D = utils.edm_torch.compute_euc_dist_square(P_const, P_var)

        return D

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        pass