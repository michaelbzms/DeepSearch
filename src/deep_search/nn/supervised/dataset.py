from typing import Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class C4Dataset(Dataset):
    def __init__(self, file: str, deserialize_fn: Callable[[np.ndarray], np.ndarray]):
        self.data: pd.DataFrame = pd.read_hdf(file)
        self.deserialize_fn = deserialize_fn

    def __getitem__(self, index):
        item = self.data.iloc[index]
        return torch.FloatTensor(self.deserialize_fn(item['board'])), float(item['target'])

    def __len__(self):
        return len(self.data)
