import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,input_features):
        super().__init__()
        # The actural multilayer neural network
        self.network = nn.Sequential(
            # Layer 0
            nn.Linear(input_features, 8, bias = False ),
            nn.ReLU(),
            # nn.Tanh(),
            # Layer 1
            nn.Linear(8,4, bias = False ),
            nn.ReLU(),
            # nn.Tanh(),
            # Layer 2
            nn.Linear(4,1, bias = False ),
        )
    def forward(self,x):
        return self.network(x)
