from os.path import join, exists
import json
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class DummyDataset(Dataset):
    def __init__(self, 
        num_batches=1
    ):
        super().__init__()
        
        self.num_batches = num_batches
        
    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return idx