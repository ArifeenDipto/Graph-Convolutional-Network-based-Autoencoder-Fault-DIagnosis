# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:19:04 2024

@author: mursh
"""

import os
import sys
import yaml
import torch
import numpy as np
import torch.nn as nn
from torch import optim

# Add the project path to the Python path
sys.path.append('D:/ML/FaultDiagnosis/src/faultdiagdip94')

from models.GCNVAE import gcnVAE

yaml_file = 'D:/ML/FaultDiagnosis/src/faultdiagdip94/utils/config.yaml'
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)


class GCNVAETrainer:
    def __init__(self):
        
        # Initialize model
        self.model = gcnVAE()
        self.model.train()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['model_params']['learning_rate'],
            amsgrad=True
        )       
        # Loss tracker
        self.training_losses = []

    @staticmethod
    def vae_loss(recon_x, x, z):
        """Custom VAE loss function."""
        reconstruction_function = nn.MSELoss()  # Reconstruction loss
        bce = reconstruction_function(recon_x, x)

        # KL Divergence
        p_mean = 0.5
        p = torch.sigmoid(z)
        p = torch.mean(p, 1)
        kld = p_mean * torch.log(p_mean / p) + (1 - p_mean) * torch.log((1 - p_mean) / (1 - p))
        kld = torch.mean(kld, 0)
        return bce + kld

    def train(self, X_tr, edge_indices, edge_weights):
        """
        Train the GCNVAE model.
        Args:
            X_tr: Input features (numpy array or tensor).
            edge_indices: Edge indices for the graph.
            edge_weights: Edge weights for the graph.
        """
        # Convert data to PyTorch tensors
        X_tr = torch.tensor(np.array(X_tr), dtype=torch.float)

        # Training loop
        for epoch in range(config['model_params']['epoch']):
            self.optimizer.zero_grad()
            X_pr, z_par = self.model(X_tr, edge_indices, edge_weights)
            loss = self.vae_loss(X_pr, X_tr, z_par)
            loss.backward()
            self.optimizer.step()

            self.training_losses.append(loss.item())
            print(f"Epoch {epoch + 1} | Loss: {loss:.4f}")

    def save_model(self):
        """Save the trained model."""
        file_name = 'gcn_vae.pkl'
        save_path = os.path.join(config['paths']['saved_model'], file_name)
        os.makedirs(config['paths']['saved_model'], exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self):
        """Load the trained model."""
        file_name = 'gcn_vae.pkl'
        load_path = os.path.join(config['paths']['saved_model'], file_name)
        
        model = gcnVAE()
        return model.load_state_dict(torch.load(load_path))


# =============================================================================
# 
# import os
# import sys
# sys.path.append('D:/ML/FaultDiagnosis/src/faultdiagdip94')
# import yaml
# import torch
# import numpy as np
# import torch.nn as nn
# from torch import optim
# 
# from models.GCNVAE import gcnVAE
# 
# yaml_file = 'D:/ML/FaultDiagnosis/src/faultdiagdip94/utils/config.yaml'
# with open(yaml_file, 'r') as file:
#     config = yaml.safe_load(file)
#     
# 
# def VAEloss(recon_x, x, z):
#     reconstruction_function = nn.MSELoss()  # mse loss
#     BCE = reconstruction_function(recon_x, x)
#     pmean = 0.5
#     p = torch.sigmoid(z)
#     p = torch.mean(p, 1)
#     KLD = pmean * torch.log(pmean / p) + (1 - pmean) * torch.log((1 - pmean) / (1 - p))
#     KLD = torch.mean(KLD, 0)
#     return BCE + KLD
# 
# model = gcnVAE()
# model.train()
# optimizer = optim.Adam(model.parameters(), lr=config['model_params']['learning_rate'], amsgrad=True)
# 
# def train(X_tr, edge_indices, edge_weights):
#     X_tr = torch.tensor(np.array(X_tr), dtype=torch.float)
#     tr_loss = []
#     for i in range(config['model_params']['epoch']):
#         optimizer.zero_grad()
#         X_pr, z_par = model(X_tr, edge_indices, edge_weights)
#         loss = VAEloss(X_pr, X_tr, z_par)
#         loss.backward()
#         optimizer.step()
#         tr_loss.append(loss.item())
#         print("Epoch %d | Loss: %.4f" %(i, loss))
# 
# def save_model():
#     file_name = 'gcn_vae.pkl'
#     torch.save(model.state_dict(), os.path.join(config['paths']['saved_model'], file_name))
# 
# =============================================================================
