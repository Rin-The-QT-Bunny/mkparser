import torch
import torch.nn as nn

import networkx as nx

class token_attention(nn.Module):
    def __init__(self,k_dim,q_dim,v_dim,hidden_dim=132):
        super().__init__()
    
    def forward(self,x):return x

class Decoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        