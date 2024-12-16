# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:18:50 2024

@author: mursh
"""

import yaml
import os
import sys
import pandas as pd
import numpy as np
from numpy import dot
from scipy import signal, stats
from numpy.linalg import norm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append('D:/ML/FaultDiagnosis/src/faultdiagdip94')

yaml_file = 'D:/ML/FaultDiagnosis/src/faultdiagdip94/utils/config.yaml'
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)


class DataProcessor:
    def __init__(self):
        
        self.folder_path = config['paths']['raw_data']
        self.save_path = config['paths']['proc_data']
        self.samples = config['params']['samples']
        self.drop = config['params']['drop']
        self.window_size = config['conv_filter']['window_size']
        
        self.data_list = []
        self.conv_data = []
                
    def data_read(self):     
        for file in sorted(os.listdir(self.folder_path)):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(self.folder_path, file))
                    df = df.drop([self.drop], axis=1)
                    df = df.iloc[:self.samples, :]
                    self.data_list.append(df)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    
    # Apply convolution filter for reducing noise
    def apply_convolution(self, sig):
        conv = np.repeat([0., 1., 0.], self.window_size)
        filtered = signal.convolve(sig, conv, mode='same') / self.window_size
        return filtered
    # Apply boxplot for removing outliers
    def box_outlier(self, data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1    #IQR is interquartile range.
        lower_bound =Q1 - 1.5 * IQR
        upper_bound =Q3 + 1.5 *IQR
        indices_lower = np.where(data <= lower_bound)[0] 
        indices_upper = np.where(data >= upper_bound)[0]
        data.drop(index = indices_lower, inplace=True)
        data.drop(index = indices_upper, inplace=True)
        return data  
        
    def reduce_noise(self):
        for i in range(len(self.data_list)):
            noise_free = self.data_list[i].apply(lambda srs: self.apply_convolution(srs))
            self.conv_data.append(noise_free)
      
    def remove_outliers(self):                
        for i in range(len(self.conv_data)):
            for c in self.conv_data[0].columns:
                new_data = self.box_outlier(self.conv_data[i][c])
                self.conv_data[i][c] = new_data
                
        for i in range(len(self.conv_data)):
            self.conv_data[i].dropna(inplace=True)
    
    def save_data(self):
        for i in range(len(self.conv_data)):
            file_name = 'data_{}.csv'.format(i)
            self.conv_data[i].to_csv(os.path.join(self.save_path, file_name))
    
    
    def processor_pipeline(self):
        
        print("Starting data processing pipeline...")
        self.data_read()
        print("Data reading complete.")
        
        self.reduce_noise()
        print("Noise reduction complete.")
        
        self.remove_outliers()
        print("Outlier removal complete.")
        
        self.save_data()
        print("Processed data saved to:", self.save_path)