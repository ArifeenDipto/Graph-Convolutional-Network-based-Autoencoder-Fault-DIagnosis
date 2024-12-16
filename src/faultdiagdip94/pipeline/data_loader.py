# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:18:34 2024

@author: mursh
"""

import sys
import os
import yaml
import torch
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
sys.path.append('D:/ML/FaultDiagnosis/src/faultdiagdip94')
#from data_handling.data_processing import DataProcessor


yaml_file = 'D:/ML/FaultDiagnosis/src/faultdiagdip94/utils/config.yaml'
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)
    
class DataPrepare:
    def __init__(self):
        
        self.columns = config['params']['columns']
        self.data_path = config['paths']['proc_data']
        self.start_idx_tr = config['params']['tr_start_index']
        self.end_idx_tr = config['params']['tr_end_index']
        self.start_idx_val = config['params']['val_start_index']
        self.end_idx_val = config['params']['val_end_index']
        self.start_idx_te = config['params']['te_start_index']
        self.end_idx_te   = config['params']['te_end_index']
        self.start_idx_f = config['params']['f_start_index'] 
        self.end_idx_f = config['params']['f_end_index']
        self.sim_mat = config['params']['similarity_metrics']
        self.data = []
        self.fault_data = []
        self.data_tr = None
        self.data_val = None
        self.data_te = None
        self.graph_data_tr = None
        self.graph_data_te =None
        self.graph_data_val =None
        
    def data_read(self):                   
        for file in sorted(os.listdir(self.data_path)):
            data_file = pd.read_csv(os.path.join(self.data_path, file))
            data_file = data_file.drop(['Unnamed: 0'], axis=1)
            self.data.append(data_file)
        
        
    def data_scaler(self, data_type):
        self.data_type = data_type        
        scaler = StandardScaler()
        scaler.fit(self.data[0])
        scaled_data = scaler.transform(self.data[0])
        scaled_data = pd.DataFrame(scaled_data, columns = self.data[0].columns)
        
        if self.data_type == 'train':     
            self.data_tr = scaled_data.iloc[self.start_idx_tr:self.end_idx_tr, :]        
        elif self.data_type == 'val':
            self.data_val = scaled_data.iloc[self.start_idx_val:self.end_idx_val, :]  
        elif self.data_type == 'test':
            self.data_te = scaled_data.iloc[self.start_idx_te:self.end_idx_te, :]  
        elif self.data_type == 'fault':
            for i in range(1, 8):
                fdata_scaled = scaler.transform(self.data[i])
                fdata_split = fdata_scaled[self.start_idx_f:self.end_idx_f, :]
                fdata_tensor = torch.tensor(fdata_split, dtype=torch.float)
                self.fault_data.append(fdata_tensor)               
            #return fault_data
        #else:
            #return print('data type is invalid')
        
    def graph_attr(self, data):
        dim = data.shape[1]
        edge = [[], []]
        for j in range(dim):
            for k in range(dim):
                if j != k:
                    edge[0].append(j)
                    edge[1].append(k)
                    
        edge_index = torch.tensor(edge, dtype=torch.long)
        
        sim_val = []
        cols = data.columns
        for c1 in range(len(cols)):
            vec_a = np.array(data[cols[c1]])
            for c2 in range(len(cols)):
                if c2!=c1:
                    vec_b = np.array(data[cols[c2]])
                    if self.sim_mat == 'Cosine':    
                        cos_sim = np.abs(dot(vec_a, vec_b)/(norm(vec_a)*norm(vec_b)))
                        sim_val.append(cos_sim)
                    else:
                        euclidean = np.linalg.norm(vec_a-vec_b)
                        sim_val.append(euclidean)
                        
        edge_weight = torch.tensor(sim_val, dtype=torch.float)
        
        return edge_index, edge_weight
    
    def graph_attributes(self, type_data):
        self.type_data = type_data
        if self.type_data == 'train':
            self.graph_data_tr = self.graph_attr(self.data_tr)
        elif self.type_data == 'val':
            self.graph_data_val = self.graph_attr(self.data_val)
        elif self.type_data == 'test':
            self.graph_data_te = self.graph_attr(self.data_te)
            
            
            
            
            
            
            