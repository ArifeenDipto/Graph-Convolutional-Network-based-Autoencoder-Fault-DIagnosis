Fault Detection and Diagnosis

This project shows a Graph Convolutional Network enabled Variational Autoencoder
for detecting and diagnosing faults in solar or PV array dataset.

Features:

1. Detects faults
2. Diagnose the root cause of the fault
3. Eliminates outlier and noise from raw dataset

Technologies used:

1.Pytorch
2.SKlearn
3.Yaml


Folder hierarchy:

Folder PATH listing for volume WorkDrive
Volume serial number is 70DA-0C15
D:.
ª   LICENSE
ª   pyproject.toml
ª   README.md
ª   structure.txt
ª   
+---src
    +---faultdiagdip94
        ª   main.py
        ª   __init__.py
        ª   
        +---data
        ª   +---processed
        ª   ª       data_0.csv
        ª   ª       data_1.csv
        ª   ª       data_2.csv
        ª   ª       data_3.csv
        ª   ª       data_4.csv
        ª   ª       data_5.csv
        ª   ª       data_6.csv
        ª   ª       data_7.csv
        ª   ª       
        ª   +---raw
        ª           F0L.csv
        ª           F1L.csv
        ª           F2L.csv
        ª           F3L.csv
        ª           F4L.csv
        ª           F5L.csv
        ª           F6L.csv
        ª           F7L.csv
        ª           
        +---models
        ª   ª   GCNVAE.py
        ª   ª   GCNVAEH3.py
        ª   ª   GCNVAEWS.py
        ª   ª   __init__.py
        ª   ª   
        ª   +---__pycache__
        ª           GCNVAE.cpython-311.pyc
        ª           __init__.cpython-311.pyc
        ª           
        +---pipeline
        ª   ª   data_loader.py
        ª   ª   data_processor.py
        ª   ª   model_tester.py
        ª   ª   model_trainer.py
        ª   ª   __init__.py
        ª   ª   
        ª   +---__pycache__
        ª           data_loader.cpython-311.pyc
        ª           data_processor.cpython-311.pyc
        ª           model_tester.cpython-311.pyc
        ª           model_trainer.cpython-311.pyc
        ª           __init__.cpython-311.pyc
        ª           
        +---saved_model
        ª       gcn_vae.pkl
        ª       __init__.py
        ª       
        +---utils
            ª   config.yaml
            ª   evaluator.py
            ª   __init__.py
            ª   
            +---__pycache__
                    evaluator.cpython-311.pyc
                    __init__.cpython-311.pyc
                    
