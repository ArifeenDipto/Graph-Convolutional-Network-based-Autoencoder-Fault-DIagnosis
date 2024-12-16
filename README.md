Fault Detection and Diagnosis

This project shows a Graph Convolutional Network enabled Variational Autoencoder
for detecting and diagnosing faults in solar or PV array dataset.

This work is published in MDPI Machines journal: 
Title: Graph-Variational Convolutional Autoencoder-Based Fault Detection and Diagnosis for Photovoltaic Arrays
Link: https://www.mdpi.com/2075-1702/12/12/894

Features:

1. Detects faults
2. Diagnose the root cause of the fault
3. Eliminates outlier and noise from raw dataset

Technologies used:

1.Pytorch
2.SKlearn
3.Yaml


Folder hierarchy:

D:.
|   folder.txt
|   LICENSE
|   pyproject.toml
|   README.md
|   structure.docx
|   structure.txt
|   
+---dist
|       faultdiagdip94-0.0.1-py3-none-any.whl
|       faultdiagdip94-0.0.1.tar.gz
|       
\---src
    +---faultdiagdip94
    |   |   main.py
    |   |   __init__.py
    |   |   
    |   +---data
    |   |   +---processed
    |   |   |       data_0.csv
    |   |   |       data_1.csv
    |   |   |       data_2.csv
    |   |   |       data_3.csv
    |   |   |       data_4.csv
    |   |   |       data_5.csv
    |   |   |       data_6.csv
    |   |   |       data_7.csv
    |   |   |       
    |   |   \---raw
    |   |           F0L.csv
    |   |           F1L.csv
    |   |           F2L.csv
    |   |           F3L.csv
    |   |           F4L.csv
    |   |           F5L.csv
    |   |           F6L.csv
    |   |           F7L.csv
    |   |           
    |   +---models
    |   |   |   GCNVAE.py
    |   |   |   GCNVAEH3.py
    |   |   |   GCNVAEWS.py
    |   |   |   __init__.py
    |   |   |   
    |   |   \---__pycache__
    |   |           GCNVAE.cpython-311.pyc
    |   |           __init__.cpython-311.pyc
    |   |           
    |   +---pipeline
    |   |   |   data_loader.py
    |   |   |   data_processor.py
    |   |   |   model_tester.py
    |   |   |   model_trainer.py
    |   |   |   __init__.py
    |   |   |   
    |   |   \---__pycache__
    |   |           data_loader.cpython-311.pyc
    |   |           data_processor.cpython-311.pyc
    |   |           model_tester.cpython-311.pyc
    |   |           model_trainer.cpython-311.pyc
    |   |           __init__.cpython-311.pyc
    |   |           
    |   +---saved_model
    |   |       gcn_vae.pkl
    |   |       __init__.py
    |   |       
    |   \---utils
    |       |   config.yaml
    |       |   evaluator.py
    |       |   __init__.py
    |       |   
    |       \---__pycache__
    |               evaluator.cpython-311.pyc
    |               __init__.cpython-311.pyc
    |               
    \---faultdiagdip94.egg-info
            dependency_links.txt
            PKG-INFO
            SOURCES.txt
            top_level.txt
