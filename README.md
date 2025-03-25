# **JESTR: Joint Embedding Space Technique for Ranking Candidate Molecules for the Annotation of Untargeted Metabolomics Data**

This repository contains the python code to train and test the JESTR model. It also contains the NPLIB1 dataset.

# **Environment Setup**
The python packages required for JESTR are given in jestr_requirements.txt. Please set up the environment as per this file using [conda](http://docs.condi.ioen/latest/)/[pip](https://pip.pypa.io/en/stable/cli/pip_install/).
All code runs under the [PyTorch framework](https://pytorch.org). The code and the models have been tested on the package versions mentioned in the jestr_requirements.txt file, but it is likely the code will work on newer versions of the packages as well.
We have released the dataset and pretrained weights for the NPLIB1 dataset. The other datasets are under licensing agreements that prohibit their public
release. If the user wants to test other datasets, they need to ensure the datasets are available to them under appropriate licenses. The model was trained and tested on GPU nVidia A100 with CUDA 11.8. The released weights are also for GPU trained models. Please ensure that the environment is set up for GPU.

# **Usage**
## **Ranking demo**
To use the pretrained model and rank a target molecule against its candidates on a given spectrum, please run the code in the notebook [JESTR.ipynb](https://github.com/HassounLab/JESTR1/blob/main/JESTR.ipynb). This notebook uses utility functions from python scripts explained below and data from [NPLIB1](https://github.com/HassounLab/JESTR1/tree/main/data/NPLIB1). The user can create their own test sets on the lines of the code given in this notebook.

## **Ranking and Training**
To rank candidates for the complete NPLIB1 dataset, use the command: 

python cand_rank_canopus.py.

To train the model from scratch, use the command: 

python train.py

For ranking candidates for other datasets, the user can prepare the dataset using the appropriate data preparation files explained below

# **License**
This project is licensed under the MIT license

# **Additional Details**
## **Data Preparation**
A description of the supplied data files is given below. The NPLIB1 dataset is released through these files.

- data_dict.pkl - a dictionary of spectra information indexed by spectra id. Each entry is a dictionary for the spectra which contains the m/z amd intensity arrays, metadata and molecule SMILES and InchiKey
- split.pkl - a dictionary containing the train/valid/test split. These are lists of inchikeys under keys 'train', 'valid', 'test'
- inchi_to_id_dict.pkl - dictionary indexed by module InChiKeys and containing a list of spectra ids belonging to that molecule
- mol_dict.pkl - a dictionary containing mapping from molecule InchiKey to rdkit mol strcuture for that molecule
- molgraph_dict.pkl - a dictionary containing mapping from molecule InchiKey to DGL graph for that molecule. This is for runtime optimization. If not provided this will be created on the fly
- cand_dict.pkl - dictionary keys by target molecule InchiKey. Each entry is a list of candidates for that molecule downloaded from PubChem. This dictionary is for the test set
- cand_dict_train_updated.pkl - same as above for the training set to be used during regularization. If you are not training with regularization, you do not need this file and can set the appropriate parameter to False in config file
- inchi_to_id_dict_wneg.pkl - dictionary containing both positive and negative (random) spectra id for each test molecule InchiKey

For the NPLIB1 dataset, some files are very large to be checked in. These are available on zenodo and the instructions for download are in the corresponding data/NPLIB1/*.txt files

## **Configuration file**

This file is called params.yaml. The parameters set in this file are:
- exp: dataset to be used. If you create your own dataset, you need to update utils.py to load it
- batch_size* - parameters to set the batch sizes during training and test
- num_epochs* - set number of epochs for contrastive and MLP training
- aug_cands - whether to do regularization or not
- contr_lr, final_lr - learning rates
- early_stopping* - epoch count for early stopping
- pretrained* - pretrained model weight path

## **Python scripts**
A description of the supplied python files is:
- train_contr.py - script to do contrastive training
- train.py - script to do final MLP training
- cand_rank_canopus.py - run ranking for the NPLIB1 dataset. For your own dataset you can create a file based on this file
- utils.py - utilities
- dataset.py - load and preprocess dataset
- models.py - all the model classes

