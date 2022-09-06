import torch
import torch.nn as nn

class SeqEncoder(nn.Module):
    def __init__(self,vector_dim,hidden_dim):
        super().__init__(self)
    def forward(self,x):
        """
        x is the input sequence with the shape of [b,t,vx]
        output has the shape of [b,vy]
        """
        return x