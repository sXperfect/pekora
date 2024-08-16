from abc import ABC, abstractmethod
import numpy as np
import torch

class BaseBinaryOP(ABC):
    def __call__(self, input, target):
        if isinstance(input, np.ndarray):
            loss = self._call_numpy(input, target)
        elif isinstance(input, torch.Tensor):
            loss = self._call_torch(input, target)
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor")

        return loss
    
    @abstractmethod
    def _call_numpy(self, input, target):
        pass
    
    @abstractmethod
    def _call_torch(self, input, target):
        pass