import torch 
from torch import nn 

class MLP(nn.Module):
    def __init__(self,input_features):
        super().__init__()
        # The actural multilayer neural network
        self.network = nn.Sequential(
            # Layer 0
            nn.Linear(input_features, 32),
            nn.ReLU(),
            # Layer 1
            nn.Linear(32,16),
            nn.ReLU(),
            # Layer 2 
            nn.Linear(16,1),
        )
    def forward(self,x):
        return self.network(x)