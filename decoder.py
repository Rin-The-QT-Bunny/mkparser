import torch
import torch.nn as nn
import numpy as np
import networkx as nx

from moic.data_structure import *
from moic.mklearn.nn.functional_net import *

class token_attention(nn.Module):
    def __init__(self,k_dim,q_dim,v_dim,hidden_dim=132):
        super().__init__()
    
    def forward(self,x):return x

class ContextFreeGrammar(nn.Module):
    def __init__(self,rule_diction,dim):
        super().__init__()
        self.cfg_diction = rule_diction
        self.arg_diction = {}
        for key in self.cfg_diction:
            input_status = self.cfg_diction[key]["input_types"]
            if (input_status is not None and len(input_status)!=0):
                L = len(self.cfg_diction[key]["input_types"])
                self.arg_diction[key] = nn.Parameter(torch.randn([L,dim]))

class Decoder(nn.Module):
    def __init__(self,s_dim,k_dim,latent_dim,CFG):
        super().__init__()
        self.cfg = CFG
        self.cfg_enabled = True
        self.counter = 0
        self.monte_carlo_enabled = False
        self.key_proj = FCBlock(132,4,k_dim,latent_dim)
        self.signal_proj = FCBlock(132,4,s_dim,latent_dim)
        self.repeater = FCBlock(132,3,s_dim+s_dim,s_dim)
        self.logprob = 0
    def forward(self,x,token_features,keys,dfs_seq = None):
        start_node = FuncNode("root")
        self.counter = 0;self.logprob = 0
        def parse_node(node,s):
            # input the current processing node and the semantics vector
            K = self.key_proj(token_features);S = self.signal_proj(s)
            # [N,z] [1,z]
            operator_distribution =  torch.softmax(9*torch.cosine_similarity(K,S),0)
            # get the pdf of operators
            if (dfs_seq != None):
                dfs_index =keys.index(dfs_seq[self.counter])
                token_chosen = keys[dfs_index]
                self.counter += 1
                self.logprob -= torch.log(operator_distribution[dfs_index])
            elif (self.monte_carlo_enabled):
                # choose the one with the maximum prob
                max_index = np.argmax(operator_distribution.detach().numpy())
                token_chosen = keys[max_index]
                self.logprob -= torch.log(operator_distribution[max_index])
            else:
                # or choose the one with the maximum probablity
                token_chosen = np.random.choice(keys,p=operator_distribution.detach().numpy())
                self.logprob -= torch.log(operator_distribution[keys.index(token_chosen)])
            # pass on the information given the conditon of op chosen and current signal
            inputs_status = self.cfg.cfg_diction[token_chosen]
            current_node = FuncNode(token_chosen)
            node.children.append(current_node)
            if (inputs_status["input_types"] is not None):
                if (len(inputs_status["input_types"])==0):
                    current_node.function = True; # function in the form of scene()
                else:
                    arg_features = self.cfg.arg_diction[token_chosen]
                    num_args = arg_features.shape[0]
                    for i in range(num_args):
                        continue_signal = self.repeater(torch.cat([s,arg_features[i:i+1,:]],-1))
                        parse_node(current_node,continue_signal)
            else:
                current_node.function = False # constants in the form of a,b,1
        parse_node(start_node,x) # parse the root node to generate a program
        return start_node.children[0],self.logprob

cfg_diction = {"+":{"output_type":"int","input_types":["int","int"]},
               "1":{"output_type":"int","input_types":None},
               "2":{"output_type":"int","input_types":None}}

cfg = ContextFreeGrammar(cfg_diction,32)
model = Decoder(32,42,132,cfg)

token_features = torch.randn([3,42])
input_signal = torch.randn([1,32])



optim = torch.optim.Adam(model.parameters(),lr=2e-3)
for epoch in range(1000):
    optim.zero_grad()
    p,l = model(input_signal,token_features,["+","1","2"],["+","+","1","2","+","2","1"])
    l.backward()
    optim.step()
    if (epoch%100==0):
        print(p,l)

model.monte_carlo_enabled = False
p,l = model(input_signal,token_features,["+","1","2"])
print(p)