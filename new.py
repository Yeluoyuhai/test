
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms
import ignite
from ignite.engine import create_supervised_evaluator

#import backdoor_attack as bd
import backdoor_attack.plot_util
import net

#%%

class Dataset_from_array(torch.utils.data.Dataset):
    def __init__(self, x, t, transform=None):
        self.transform = transform
        self.x = x
        self.t = t
        self.num = x.shape[0]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        x = self.x[idx]
        t = self.t[idx]

        if not self.transform is None:
            x = self.transform(x)

        return x, t
# that's all
# ok ok
