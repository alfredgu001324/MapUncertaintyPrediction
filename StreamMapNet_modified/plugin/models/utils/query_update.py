import math
import torch
import torch.nn as nn 
import numpy as np
from mmcv.cnn import bias_init_with_prob, xavier_init

class MotionMLP(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=512, identity=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.identity = identity

        self.fc = nn.Sequential(
            nn.Linear(c_dim + f_dim, 2*f_dim),
            nn.LayerNorm(2*f_dim),
            nn.ReLU(),
            nn.Linear(2*f_dim, f_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.fc:
            for param in m.parameters():
                if param.dim() > 1:
                    if self.identity:
                        nn.init.zeros_(param)
                    else:
                        nn.init.xavier_uniform_(param)

    def forward(self, x, c):
        xc = torch.cat([x, c], dim=-1)
        out = self.fc(xc)

        if self.identity:
            out = out + x
        
        return out
