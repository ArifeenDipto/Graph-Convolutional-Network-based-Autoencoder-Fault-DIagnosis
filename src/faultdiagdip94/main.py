# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:11:36 2024

@author: mursh
"""
import sys
import yaml
sys.path.append('D:/ML/FaultDiagnosis/src/faultdiagdip94')
from pipeline.data_loader import DataPrepare
from pipeline.data_processor import DataProcessor
from pipeline.model_trainer import GCNVAETrainer
from pipeline.model_tester import evaluate_function
from models.GCNVAE import gcnVAE
from utils.evaluator import Evaluator

import torch
import numpy as np

yaml_file = 'D:/ML/FaultDiagnosis/src/faultdiagdip94/utils/config.yaml'
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

## Processing raw data to eliminate noise and outliers
data_processor = DataProcessor()
data_processor.processor_pipeline()


## Loading the train, validation and test pre processed 
## data and train the model

data_prepare = DataPrepare()
data_prepare.data_read()

## Scaling the loaded datasets

data_prepare.data_scaler('train')
train_data = data_prepare.data_tr
data_prepare.data_scaler('val')
val_data = data_prepare.data_val
data_prepare.data_scaler('test')
test_data = data_prepare.data_te
data_prepare.data_scaler('fault')
fault_data = data_prepare.fault_data

## Generating graph attributes (edge indices and weights)

data_prepare.graph_attributes('train')
train_graph = data_prepare.graph_data_tr
data_prepare.graph_attributes('val')
val_graph = data_prepare.graph_data_val
data_prepare.graph_attributes('test')
test_graph = data_prepare.graph_data_te


## Model training
trainer = GCNVAETrainer()
trainer.train(train_data, train_graph[0], train_graph[1])

## Saving the trained model
trainer.save_model()

## Loading the saved model
#model = trainer.load_model()

model = gcnVAE()
model.load_state_dict(torch.load(config['paths']['saved_model']))
model = model.eval()

result = evaluate_function('val', model, val_data, train_graph[0], train_graph[1]) 

fr = evaluate_function('fault', model, fault_data, train_graph[0], train_graph[1])

