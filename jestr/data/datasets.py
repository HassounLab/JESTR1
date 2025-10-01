import pandas as pd
import json
import typing as T
import numpy as np
import torch
import massspecgym.utils as utils
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
import dgl
from collections import defaultdict
from massspecgym.data.transforms import SpecTransform, MolTransform, MolToInChIKey
from massspecgym.data.datasets import MassSpecDataset
import jestr.utils.data as data_utils
from torch.nn.utils.rnn import pad_sequence
from massspecgym.models.base import Stage
import pickle
import math
import itertools
from rdkit.Chem import AllChem
from rdkit import Chem
class JESTR1_MassSpecDataset(MassSpecDataset):
    def __init__(
        self,
        spectra_view: str,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.spectra_view = spectra_view

    def __getitem__(self, i, transform_spec: bool = True, transform_mol: bool = True):

        spec = self.spectra[i]
        metadata = self.metadata.iloc[i]
        mol = metadata["smiles"]

        # Apply all transformations to the spectrum
        item = {}
        if transform_spec and self.spec_transform:
            if isinstance(self.spec_transform, dict):
                for key, transform in self.spec_transform.items():
                    item[key] = transform(spec) if transform is not None else spec
            else:
                item["spec"] = self.spec_transform(spec)
        else:
            item["spec"] = spec

        if self.return_mol_freq:
            item["mol_freq"] = metadata["mol_freq"]

        if self.return_identifier:
            item["identifier"] = metadata["identifier"]
            
        # Apply all transformations to the molecule
        if transform_mol and self.mol_transform:
            if isinstance(self.mol_transform, dict):
                for key, transform in self.mol_transform.items():
                    item[key] = transform(mol) if transform is not None else mol
            else:
                item["mol"] = self.mol_transform(mol)
        else:
            item["mol"] = mol
        return item

class ContrastiveDataset(Dataset):
    def __init__(
        self,
        spec_mol_data,
    ):
        super().__init__()
    
        indices = spec_mol_data.indices
        self.spec_mol_data = spec_mol_data
        self.smiles_to_specmol_ids = spec_mol_data.dataset.metadata.loc[indices].groupby('smiles').indices
        self.smiles_to_spec_couter = defaultdict(int)
        self.smiles_list = list(self.smiles_to_specmol_ids.keys())

    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, i:int) -> dict:
        mol = self.smiles_list[i]

        # select spectrum (iterate through list of spectra)
        specmol_ids = self.smiles_to_specmol_ids[mol]
        counter = self.smiles_to_spec_couter[mol]
        specmol_id = specmol_ids[counter % len(specmol_ids)]

        item = self.spec_mol_data.__getitem__(specmol_id)
        self.smiles_to_spec_couter[mol] = counter+1
        # item['smiles'] = mol
        # item['spec_id'] = specmol_id
        return item

    @staticmethod
    def collate_fn(batch: T.Iterable[dict], spec_enc: str, spectra_view: str, stage=None) -> dict:
        mol_key = 'cand' if stage == Stage.TEST else 'mol'
        non_standard_collate = ['mol', 'cand']
        require_pad = False

        collated_batch = {}
        # standard collate
        for k in batch[0].keys():
            if k not in non_standard_collate:
                collated_batch[k] = default_collate([item[k] for item in batch])
                
        # batch graphs
        batch_mol = []
        batch_mol_nodes= []

        for item in batch:
            batch_mol.append(item[mol_key])
            batch_mol_nodes.append(item[mol_key].num_nodes())

        collated_batch[mol_key] = dgl.batch(batch_mol)
        collated_batch['mol_n_nodes'] = batch_mol_nodes

        return collated_batch
    
 

class ExpandedRetrievalDataset:
    '''Used for testing only 
    Assumes 'fold' column defines the split'''
    def __init__(self,
                 mol_label_transform: MolTransform = MolToInChIKey(),
                 candidates_pth: T.Optional[T.Union[Path, str]] = None,
                 candidate_file_key: str = "smiles",
                **kwargs):
        
        self.instance = JESTR1_MassSpecDataset(**kwargs, return_mol_freq=False)
        # super().__init__(**kwargs)

        self.candidates_pth = candidates_pth
        self.mol_label_transform = mol_label_transform
        
        # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
        with open(self.candidates_pth, "r") as file:
            candidates = json.load(file)

        self.candidates = {}
        for s, cand in candidates.items():
            self.candidates[s] = [c for c in cand if '.' not in c]
        
        self.spec_cand = [] #(spec index, cand_smiles, true_label)

        if 'smiles' not in self.metadata.columns or candidate_file_key == "identifierss":
            if not isinstance(self.metadata.iloc[0]['identifier'], str):
                self.metadata['smiles'] = self.metadata['identifier'].apply(str)
            else:
                self.metadata['smiles'] = self.metadata['identifier']

        test_smiles = self.metadata[self.metadata['fold'] == "test"]['smiles'].tolist()
        test_ms_id = self.metadata[self.metadata['fold'] == "test"]['identifier'].tolist()
        
        spec_id_to_index = dict(zip(self.metadata['identifier'], self.metadata.index))
        for spec_id, s in zip(test_ms_id, test_smiles):
            candidates = self.candidates[s]
            # mol_label = self.mol_label_transform(s)
            # labels = [self.mol_label_transform(c) == mol_label for c in candidates]
            labels = [c == s for c in candidates]
            if len(candidates) == 0:
                print(f"Skipping {spec_id}; empty candidate set")
                continue
            # if not any(labels):
            #     print(f"Target smiles not in candidate set")


            self.spec_cand.extend([(spec_id_to_index[spec_id], candidates[j], k) for j, k in enumerate(labels)])
    
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)
    
    def __len__(self):
        return len(self.spec_cand)

    def __getitem__(self, i):
        spec_i = self.spec_cand[i][0]
        cand_smiles = self.spec_cand[i][1]
        label = self.spec_cand[i][2]

        item = self.instance.__getitem__(spec_i, transform_mol=False)
        item['cand'] = self.mol_transform(cand_smiles)
        item['cand_smiles'] = cand_smiles
        item['label'] = label
        return item