# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:19:56 2024

@author: mursh
"""


import sys
import os
import yaml
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import torch
sys.path.append('D:/ML/FaultDiagnosis/src/faultdiagdip94')

yaml_file = 'D:/ML/FaultDiagnosis/src/faultdiagdip94/utils/config.yaml'
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

class Evaluator:
    def __init__(self):
        """
        Initialize the ReconstructionEvaluator with a confidence value.
        Args:
            conf_val (float): Confidence level for interval computation (e.g., 0.95 for 95% confidence).
        """
        self.conf_val = config['params']['confidence_value']

    @staticmethod
    def recon_perf(true_data, pred_data):
        """
        Calculate reconstruction performance metrics.
        Args:
            true_data (array-like): Ground truth data.
            pred_data (array-like): Predicted data.
        Returns:
            dict: Dictionary containing MSE, MAE, and MAPE.
        """
        result = {
            'mse': mean_squared_error(true_data, pred_data),
            'mae': mean_absolute_error(true_data, pred_data),
            'mape': mean_absolute_percentage_error(true_data, pred_data),
        }
        return result

    @staticmethod
    def confidence_interval(data, conf_value):
        """
        Compute the confidence interval for the given data.
        Args:
            data (array-like): Data for which to compute the confidence interval.
            conf_value (float): Confidence level.
        Returns:
            tuple: Lower and upper bounds of the confidence interval.
        """
        mean, std = data.mean(), data.std(ddof=1)
        lower, higher = stats.norm.interval(conf_value, loc=mean, scale=std)
        return lower, higher

    def spe_control_limit(self, tr_data, pred_data):
        """
        Compute SPE (Squared Prediction Error) and its control limits.
        Args:
            tr_data (array-like): True data.
            pred_data (array-like): Predicted data.
        Returns:
            tuple: Array of SPE values, lower limit, and upper limit.
        """
        hist = []
        for i in range(tr_data.shape[0]):
            spe = np.matmul((tr_data - pred_data)[i, :].T, (tr_data - pred_data)[i, :])
            hist.append(spe)
        hist_array = np.array(hist)
        low_lim, high_lim = self.confidence_interval(hist_array, self.conf_val)
        return hist_array, low_lim, high_lim

    def eval_normal(self, model, data, edge_idx, edge_wt, eval_type):
        
        data = torch.tensor(np.array(data), dtype=torch.float)
        
        pred = model(data, edge_idx, edge_wt)
        pred_data = pred[0].detach().numpy()
        true_data = data.detach().numpy()

        recon_result = self.recon_perf(true_data, pred_data)
        error, low, high = self.spe_control_limit(true_data, pred_data)

        if eval_type == 'train':
            return error, recon_result, pred_data, true_data
        elif eval_type == 'val':
            return error, recon_result, high, pred_data, true_data
        elif eval_type == 'test':
            return error, recon_result, pred_data, true_data
        else:
            raise ValueError("Invalid evaluation type. Choose from 'train', 'val', or 'test'.")

    def eval_fault(self, model, data, edge_idx, edge_wt):
        """
        Evaluate the model's performance under faulty conditions.
        Args:
            model: Trained model.
            data: List of input data samples as PyTorch tensors.
            edge_idx: Edge indices for the graph.
            edge_wt: Edge weights for the graph.
            eval_type (str): Evaluation type ('fault').
        Returns:
            Various outputs depending on eval_type.
        """
        #if eval_type != 'fault':
            #raise ValueError("Invalid evaluation type. Use 'fault' for fault evaluation.")

        fault_pred = []
        fault_data = []
        recon_results = []
        fault_errors = []

        for i in range(len(data)):
            pred = model(data[i], edge_idx, edge_wt)
            pred_data = pred[0].detach().numpy()
            true_data = data[i].detach().numpy()

            fault_pred.append(pred_data)
            fault_data.append(true_data)

            error, _, _ = self.spe_control_limit(true_data, pred_data)
            fault_errors.append(error)
            recon_results.append(self.recon_perf(true_data, pred_data))

        return fault_errors, recon_results, fault_pred, fault_data


# =============================================================================
# def recon_perf(true_data, pred_data):
#     result = {}
#     mse = mean_squared_error(true_data, pred_data)
#     mae = mean_absolute_error(true_data, pred_data)
#     mape = mean_absolute_percentage_error(true_data, pred_data)
#              
#     result['mse'] = mse
#     result['mae'] = mae
#     result['mape'] = mape
#     
#     return result
# 
# def ConfidenceInterval(data, conf_value):
#     mean, std = data.mean(), data.std(ddof=1)
#     lower, higher = stats.norm.interval(conf_value, loc=mean, scale=std)
#     return lower, higher
# 
# def SPEControlLimit(tr_data, pred_data, conf_val):
#     hist=[]
#     for i in range(tr_data.shape[0]):
#         SPE = np.matmul((tr_data-pred_data)[i, :].T, (tr_data-pred_data)[i, :])
#         hist.append(SPE)
#     hist1 = np.array(hist)
#     low_lim, high_lim = ConfidenceInterval(hist1, conf_val)
#     return hist1, low_lim, high_lim
# 
# 
# def eval_normal(model, data, conf_val, edge_idx, edge_wt, eval_type):
#     
#     pred = model(data, edge_idx, edge_wt)
#     
#     pred1 = pred[0].detach().numpy()
#     data1 = data.detach().numpy()
#     recon_result = recon_perf(data1, pred1)
#     error, low, high = SPEControlLimit(data1, pred1, conf_val)
#     if eval_type == 'train':
#         return error, recon_result, pred1, data1
#     elif eval_type == 'val':
#         return error, recon_result, high, pred1, data1
#     elif eval_type == 'test':
#         return error, recon_result, pred1, data1
#     else:
#         return print('invalid evaluation type')
#     
# def eval_fault(model, data, conf_val, edge_idx, edge_wt, eval_type):
#     if eval_type == 'fault':
#         fault_pred = []
#         fault_data = []
#         recon_result = []
#         fault_error = []
#         for i in range(len(data)):
#             pred = model(data[i], edge_idx, edge_wt)
#             pred1 = pred[0].detach().numpy()
#             data1 = data[i].detach().numpy()
#             fault_pred.append(pred1)
#             fault_data.append(data1)
#             error,_,_ = SPEControlLimit(data1, pred1, conf_val)
#             fault_error.append(error)
#             recon_result.append(recon_perf(data1, pred1))
#         return fault_error, recon_result, fault_pred, fault_data
#             
#     else:
#         return print('invalid evaluation type')
# 
# =============================================================================
