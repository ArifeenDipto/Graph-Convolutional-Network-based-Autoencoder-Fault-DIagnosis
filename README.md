Fault Detection and Diagnosis

This project shows a Graph Convolutional Network enabled Variational Autoencoder
for detecting and diagnosing faults in solar or PV array dataset.

This work is published in MDPI Machines journal: 
Title: Graph-Variational Convolutional Autoencoder-Based Fault Detection and Diagnosis for Photovoltaic Arrays
Link: https://www.mdpi.com/2075-1702/12/12/894

Abstract:
Solar energy is a critical renewable energy source, with solar arrays or photovoltaic systems widely used to convert solar energy into electrical energy. However, solar array systems can develop faults and may exhibit poor performance. Diagnosing and resolving faults within these systems promptly is crucial to ensure reliability and efficiency in energy generation. Autoencoders and their variants have gained popularity in recent studies for detecting and diagnosing faults in solar arrays. However, traditional autoencoder models often struggle to capture the spatial and temporal relationships present in photovoltaic sensor data. This paper introduces a deep learning model that combines a graph convolutional network with a variational autoencoder to diagnose faults in solar arrays. The graph convolutional network effectively learns from spatial and temporal sensor data, significantly improving fault detection performance. We evaluated the proposed deep learning model on a recently published solar array dataset for an integrated power probability table mode. The experimental results show that the model achieves a fault detection rate exceeding 95% and outperforms the conventional autoencoder models. We also identified faulty components by analyzing the model’s reconstruction error for each feature, and we validated the analysis through the Kolmogorov–Smirnov test and noise injection techniques.

Features of the model:

1. Detects faults
2. Diagnose the root cause of the fault
3. Eliminates outlier and noise from raw dataset

Technologies used:

1.Pytorch
2.SKlearn
3.Yaml



   
