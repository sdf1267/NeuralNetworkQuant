# import polars as pl
# %%
# Utils
import numpy as np
import pandas as pd

# Machine Learning
import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, input_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return self.linear(x)
