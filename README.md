# JESTR
Joint Embedding Space Technique for Ranking Candidate Molecules for the Annotation of Untargeted Metabolomics Data

Environment:
The python packages required for JESTR are given in jestr_requirements.txt. Please set up the environment as per this file using conda/pip

We have released the dataset and pretrained weights for the NPLIB1 dataset. The other datasets are under licensing agreements that prohibit thei public
release. The user can download those data themselves and prepare them as per instructions below.

To rank candidates for the NPLIB1 dataset, use the python script:
python cand_rank_canopus.py
The user can load the supplied model weights as eplained below.

To train the model from scratch, use the python script:
pythn train.py

A description of the supplied data files is:

- data_dict.pkl - a dictionary of spectra information indexed by spectra id. Each entry is a dictionary for the spectra which contains the m/z amd intensity arrays,
metadata and molecule SMILES and InchiKey

- split.pkl - a dictonary containing the train/valid/test split. These are lists of spectra ids under keys 'train', 'valid', 'test'
  
- inchi_to_id_dict.pkl - dictionary indexed by module InChiKeys and containing a list of spectra ids belonging to that molecule
  
- mol_dict.pkl - a dictionary containing mapping from molecule InchiKey to rdkit mol strcuture for that molecule
  
- molgraph_dict.pkl - a dictionary containing mapping from molecule InchiKey to graph for that molecule. This is for optimization. If not provided this will be created on the fly

- cand_dict.pkl - dictionary keys by target molecule InchiKey. Each entry is a list of candidates for that molecule downloaded from PubChem. This dictionary is for the test set
  
- cand_dict_train_updated.pkl - same as above for the training set to be used during regularization. If you are not training for regularization, you do not need this file and can set the appropriate parameter to False in config file
  
- inchi_to_id_dict_wneg.pkl - dictionary containing both positive and negative (random) spectra id for each test molecule InchiKey

- pretrained*.pt - pretrained model weights for NPLIB1 dataset with regularization for the three models used in JESTR

For the NPLIB1 dataset, some files are very large to be checked in. These are available on zenodo and the instructions for download are in
data/NPLIB1/*.txt files

Config file: This file is called params.yaml. The parameters set in this file are:
- exp: dataset to be used. If you create your own dataset, you need to update utils.py to load it
- batch_size* - parameters to set the batch sizes during training and test
- num_epochs* - set number of epochs for contrastive and MLP training
- aug_cands - whether to do augmentation or not
- contr_lr, final_lr - learning rates
- early_stopping* - epoch count for early stopping
- pretrained* - pretrained model weight path

A description of the supplied python files is:
- train_contr.py - script to do contrastive training
- train.py - script to do final training
- cand_rank_canopus.py - run ranking for the NPLIB1 dataset. For your own dataset you can modify this file
- utils.py - utilities
- dataset.py - load and preprocess dataset
- models.py - all the model classes

