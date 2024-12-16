# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:34:50 2024

@author: mursh
"""
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv

sys.path.append('D:/ML/FaultDiagnosis/src/faultdiagdip94')
yaml_file = 'D:/ML/FaultDiagnosis/src/faultdiagdip94/utils/config.yaml'
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

class gcnVAE(torch.nn.Module):
    def __init__(self):
        super(gcnVAE, self).__init__()
        
        self.input_dim = config['model_params']['input_dim']
        self.improved = config['model_params']['improved']

        self.enc1 = GCNConv(self.input_dim, 8, improved=self.improved) #encoding layer 1
        self.bn1 = BatchNorm(8) #batch normalization layer
        
        self.enc2 = GCNConv(8, 4, improved=self.improved) #encoding layer 2
        self.bn2 =  BatchNorm(4)
        
        self.enc3 = GCNConv(4, 1, improved=self.improved) #encoding layer 2
        self.bn3 = BatchNorm(1)
        
        self.mean = nn.Linear(1, 1) #mean sampling layer
        self.logstd = nn.Linear(1, 1)#standard dev sampling layer
        
        self.dec1 = GCNConv(1, 4, improved=self.improved) # deconding layer 1
        self.bn4 = BatchNorm(4)#batch normalization layer 2
        
        self.dec2 = GCNConv(4, 8, improved=self.improved) # deconding layer 1
        self.bn5 = BatchNorm(8)#batch normalization layer 2
        
        self.dec3 = GCNConv(8, self.input_dim, improved=self.improved)# decoding layer 2
  

    def forward(self, data, edg_idx, edg_wt):
        x, edge_index, edge_weight= data, edg_idx, edg_wt #graph data, graph edges, graph edge weights
        
        x1 = self.enc1(x, edge_index, edge_weight)
        x1 = self.bn1(x1)
        x1 = F.tanh(x1)
        
        x2 = self.enc2(x1, edge_index, edge_weight)
        x2 = self.bn2(x2)
        x2 = F.tanh(x2)
        
        x3 = self.enc3(x2,  edge_index, edge_weight)
        x3 = self.bn3(x3)
        x3 = F.tanh(x3)
        
        self.hidden_vars =x3
        
        z_mean = self.mean(x3)
        z_logstd = self.logstd(x3)
        
        z = torch.exp(z_logstd) + z_mean
        self.z_val =z
        
        x4 = self.dec1(z, edge_index, edge_weight)
        x4 = self.bn4(x4)
        x4 = F.tanh(x4)      
        
        x4c = x4+x2
        
        
        x5 = self.dec2(x4c, edge_index, edge_weight)
        x5 = self.bn5(x5)
        x5 = F.tanh(x5)

        x5c = x5+x1           
        
        x6 = self.dec3(x5c, edge_index, edge_weight)
        return x6, z
