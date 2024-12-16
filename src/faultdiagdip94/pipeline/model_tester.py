# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:19:17 2024

@author: mursh
"""
import os
import sys
import yaml
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
sys.path.append('D:/ML/FaultDiagnosis/src/faultdiagdip94')

from models.GCNVAE import gcnVAE
from utils.evaluator import Evaluator

yaml_file = 'D:/ML/FaultDiagnosis/src/faultdiagdip94/utils/config.yaml'
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)


def evaluate_function(eval_type, model, data, edge_idx, edge_wt,):
    evaluate = Evaluator()
    if eval_type == 'val':
       eval_val = evaluate.eval_normal(model, data, edge_idx, edge_wt, eval_type)
       return eval_val
    elif eval_type == 'test':
       eval_test = evaluate.eval_normal(model, data, edge_idx, edge_wt, eval_type)
       return eval_test
    elif eval_type =='fault':
       eval_fault = evaluate.eval_fault(model, data, edge_idx, edge_wt)
       return eval_fault
    else:
        print('chose wrong option')
       
       
       
       
       