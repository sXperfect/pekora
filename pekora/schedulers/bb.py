import typing as t
import warnings
import torch
from torch.optim import Optimizer, lr_scheduler

class BorzelaiBorweinScheduler(lr_scheduler._LRScheduler):
    """
    Borzelai-Borwein-based Learning Rate Scheduler.
    """
    
    def __init__(
        self,
        optimizer:Optimizer,
        lr=1e-1,
        max_lr=1e-1,
        min_lr=1e-8,
        steps=1,
        beta=1e-2,
        weight_decay=0.,
        last_epoch:int=-1,
        verbose:bool=False
    ) -> None:

        assert len(optimizer.param_groups) == 1, \
            ValueError("BorzelaiBorweinScheduler doesn't support per-parameter options (parameter groups)")

        assert lr > 0.0, ValueError("Invalid initial learning rate: {}".format(lr))
        assert steps > 0, ValueError("Invalid steps: {}".format(steps))
        assert 0.0 < beta <= 1.0, ValueError("Invalid beta value: {}".format(beta))
        assert min_lr > 0.0, ValueError("Invalid minimal learning rate: {}".format(min_lr))
        assert max_lr > min_lr, ValueError("Invalid maximal learning rate: {}".format(max_lr))
        assert weight_decay >= 0.0, ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.lr = lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.steps = steps
        self.beta = beta
        self.weight_decay = weight_decay

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self._reset()
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_default_lr(self):
        return self.lr

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        optimizer = self.optimizer
        assert len(optimizer.param_groups) == 1, \
            ValueError("BB doesn't support per-parameter options (parameter groups)")
            
        group = optimizer.param_groups[0]

        if self._step_count % self.steps == 0:
            self._bb_iter += 1
            sum_dp_dg = 0
            sum_dp_norm = 0
            sum_dg_norm = 0
            
            for i, p in enumerate(group['params']):
                if p.requires_grad:
                    if self._step_count == 1:
                            self.state[i] = {
                                'grad_aver': torch.zeros_like(p),
                                'grad_prev': torch.zeros_like(p),
                                'params_prev': torch.zeros_like(p),
                            }
                            
                    if self._bb_iter > 1:
                        params_diff = p.detach() - self.state[i]['params_prev']
                        grad_diff = self.state[i]['grad_aver'] - self.state[i]['grad_prev']
                        
                        sum_dp_dg += (grad_diff * params_diff).sum().item()
                        # sum_dp_norm += params_diff.norm().item() ** 2
                        # sum_dg_norm += grad_diff.norm().item() ** 2
                        sum_dp_norm += params_diff.square().sum().item()
                        sum_dg_norm += grad_diff.square().sum().item()
                        
                    if self._bb_iter > 0:
                        self.state[i]['grad_prev'].copy_(p.grad.detach())
                        self.state[i]['params_prev'].copy_(p.detach())
                        
                        
            if self._bb_iter > 1:
                if abs(sum_dp_dg) >= 1e-10:
                    lr_hat = sum_dp_norm / (sum_dp_dg * self.steps)
                    lr_scaled = abs(lr_hat) * (self._bb_iter + 1)
                    if self.max_lr > lr_scaled > self.min_lr:
                        lr = abs(lr_hat)
                    else:
                        lr = self._last_lr[0]
                        
                    self._last_lr = [lr] * len(self.optimizer.param_groups)

        return self._last_lr

    def _reset(self):
        #? Set hidden states
        self._last_lr = [self.lr] * len(self.optimizer.param_groups)
        self._last_x_t = None
        self._last_grad_t = None
        self._curr_x_t = None
        self._curr_grad = None
        self._bb_iter = -1
        self.state = [None] * len(self.optimizer.param_groups[0]['params'])