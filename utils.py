# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 20:09:22 2023

@author: apurv
"""
import pickle
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import sys
import numpy as np
import torch
import dgl
from dgllife.utils import EarlyStopping
import random
import random
from dataset import single_molgraph_return

class Spectra_data(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        return self.data_list[idx]
    
class MultiView_data(Dataset):
    """
    Dataset that accepts data points representing for each entity one or multiple data of every view of each entity. Assumes all entity have the same set of views. Retrieves a single data of each view for all entities queried

    Parameters:
    -----------
    entity_dict : dict
        dictionary keyed by entity names (also stored in entity_list).
        values contain a list of views, each view contains a list of data for that view
        assumes that the ith view is always the same view across all entities
    entity_list : list
        name of entities in entity_dict. Order of list gives the index of the entity to query from entity_dict
    num_views : int
        number of views to get data from
    use_random : boolean
        specifies to pick example from each view randomly (when True) or iterate a step through list of data for each view when queried each time

    
    """
    def __init__(self, molgraph_dict, entity_dict, entity_list, num_views, use_random=False):
        self.entity_dict = entity_dict
        self.entity_list = entity_list
        self.num_views = num_views
        self.use_random = use_random
        self.molgraph_dict = molgraph_dict


        # run check here: num_views and formatting of data

        self.entity_counter_dict = {entity:[0 for i in range(self.num_views)] for entity in self.entity_dict}

    def __len__(self):
        """
        Dataset length defined by the number of entities not total datapoints

        Returns:
        --------
        size : int
            number of entities
        """
        return len(self.entity_list)


    def __getitem__(self, idx):
        """
        Obtains for the index of entity_list one data of each view for that entity

        Parameters:
        -----------
        idx : int
            index of entity_list entity to get views from
        
        Returns:
        --------
        data_view1 : Anything
            a data point from the first view of the selected entity
        data_view2 : Anything
            ...
        data_viewN: Anything
            ...
        """

        entity = self.entity_list[idx]
        
        if self.use_random:
            entity_views = self.entity_dict[entity]
            ret = [np.random.choice(entity_views[view_i]) for view_i in range(self.num_views)]
            return tuple(ret)

        else:
            entity_views = self.entity_dict[entity]
            entity_counters = self.entity_counter_dict[entity]
            ret = [entity_views[view_i][entity_counters[view_i]] for view_i in range(self.num_views)]
            
            # next increment each counter

            for view_i in range(self.num_views):
                entity_counters[view_i] += 1
                if entity_counters[view_i] >= len(entity_views[view_i]):
                    entity_counters[view_i] = 0

            return ret, entity

def get_ms_array(msz, transformation, max_mz, resolution):
    n_cells = int(max_mz / resolution)
    ms_array = np.zeros(n_cells, np.float32)
    mz_intensity = [p for p in msz if p[1] < max_mz + 1]
    for p in mz_intensity:
        bin_idx = int((p[1] - 1) / resolution)
        ms_array[bin_idx] += p[0]
    if transformation == "log10over3":
        out = np.log10(ms_array + 1) / 3
    else:
        out = ms_array
    return out

def get_ms_array_batch(msz, transformation, max_mz, resolution):
    bin_l = []
    lengths = np.array([len(mz) for mz in msz])
    int_b, pad = torch.tensor(0), torch.tensor(0)
    for msz_i in msz:
        bin_msz = get_ms_array(msz_i, transformation, max_mz, resolution)
        bin_l.append(bin_msz)
        
    bin_msz = np.vstack(bin_l)
    bin_msz = torch.from_numpy(bin_msz)
    return bin_msz, int_b, pad, lengths

class collate_spectra_data(object):
    # class to hold and batch graph objects
    def __init__(self, *param_l):
        self.molgraph_dict = param_l[0]
        self.params = param_l[1]
        self.mz_log_lims = (self.params['mz_log_low'], self.params['mz_log_high'])
        self.mz_spacing = self.params['mz_spacing']
        self.mz_precision = self.params['mz_precision']
        self.embd_dim = self.params['sinus_embed_dim']
        self.inter = self.params['inter']
        self.resolution = self.params['resolution']
        self.max_mz = self.params['max_mz']
        self.transformation = 'log10over3'
        if self.mz_precision is None:
            self.local_dtype = torch.float
        elif self.mz_precision == 16:
            self.local_dtype = torch.float16
        elif self.mz_precision == 32:
            self.local_dtype = torch.float32
        elif self.mz_precision == 64:
            self.local_dtype = torch.float64
        else:
            raise ValueError()
        self.spec_enc = self.params['spec_enc']
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        
    def __call__(self, batch):
        mz_l = [b[1][:,1] for b in batch]
        int_l = [b[1][:,0] for b in batch]
        if self.spec_enc == 'MLP_BIN':
            mzi = [b[1] for b in batch]
            mz_b, int_b, pad, lengths = get_ms_array_batch(mzi, self.transformation, self.max_mz, self.resolution)
        else:
            mz_b, int_b, pad, lengths = self.collate_mzi(mz_l, int_l)
        fp_l = [b[2] for b in batch]
        fp_b = np.vstack(fp_l)
        fp_b = torch.from_numpy(fp_b).to(dtype=torch.float32)
        if self.inter:
            y = [b[3] for b in batch]
            y = torch.from_numpy(np.stack(y)).long()
            #y = torch.stack(y, 0)
        else:
            y = None
        inchi_l = [b[0] for b in batch]
        batched_g = [self.molgraph_dict[b[0]] for b in batch]
        batched_g = dgl.batch(batched_g)
        ret_tuple = (batched_g, mz_b, int_b, pad, fp_b, y, lengths, inchi_l)
        return ret_tuple

    def get_sinus_repr(self, mz_tensor):
        
        ####MODIFICATION MADE HERE: MATCH EQUATION ############
        if self.mz_spacing == 'log':
            frequency = 2 * np.pi / (10**self.mz_log_lims[0]*torch.logspace(
                0, 1, int(self.embd_dim / 2), base=(10**self.mz_log_lims[1])/(10**self.mz_log_lims[0]), dtype=self.local_dtype))
    
            #print check this is correct
            #print(frequency)
            #for D = 4 should be
            # [2pi/10e-2*(10e5)^(0/2), 2pi/10e-2*(10e5)^(2/2)]
            # [2pi/10e-2, 2pi/10e3]
            # [62.831, 0.00062]
    
        #######################################################
        elif self.mz_spacing == 'linear':
            frequency = 2 * np.pi / torch.linspace(
                self.mz_log_lims[0], self.mz_log_lims[1], int(self.embd_dim / 2), dtype=self.local_dtype)
        else:
            raise ValueError('mz_spacing must be either log or linear')
    
        omega_mz = frequency.reshape((1, 1, -1)) * mz_tensor
        sin = torch.sin(omega_mz)
        cos = torch.cos(omega_mz)
    
        ######## MODIFICATION MADE HERE: sin and cos will alternate ##
        #mz_vecs = torch.cat([sin, cos], axis=2)
    
        sin = sin.unsqueeze(3)
        cos = cos.unsqueeze(3)
        mz_vecs = torch.cat([sin, cos], axis=3)
        mz_vecs = mz_vecs.view(mz_vecs.shape[0], mz_vecs.shape[1], 2*sin.shape[2])
    
        return mz_vecs
        
    def collate_mzi(self, mz_l, int_l):
        lengths = np.array([len(mz) for mz in mz_l])
        b, l = len(mz_l), max(lengths)
        #tmp_tensor = torch.tensor((), dtype=self.local_dtype)
        mz_block = np.zeros((b, l+2))
    
        for i in range(b):
            #mz_block[i, 1:len(mz_l[i])-1] = mz_l[i][1:-1]
            mz_block[i, 1:len(mz_l[i])+1] = mz_l[i]
            if self.spec_enc == 'TFM':
                mz_block[i, 0] = self.SOS
                mz_block[i, lengths[i]+1] = self.EOS
    
        mz_b = torch.from_numpy(mz_block)
        mz_b = mz_b.to(self.local_dtype)
    
        lengths = np.array([len(inte) for inte in int_l])
        b, l = len(int_l), max(lengths)
        #tmp_tensor = torch.tensor((), dtype=self.local_dtype)
    
        int_block = np.zeros((b, l+2))
    
        for i in range(b):
            int_block[i, 1:len(int_l[i])+1] = int_l[i]
            if self.spec_enc == 'TFM':
                int_block[i, 0] = self.SOS
                int_block[i, lengths[i]+1] = self.EOS
    
        int_b = torch.from_numpy(int_block)
        int_b = int_b.to(self.local_dtype)
        int_b = int_b.unsqueeze(2)
        pad = mz_b == 0
        
        mz_b = mz_b.unsqueeze(2)
        mz_b = self.get_sinus_repr(mz_b)
        
        mz_b = ~pad.unsqueeze(2) * mz_b
                
        return mz_b, int_b, pad, lengths

def get_batch_graphs(ik_list, local_molgraph_dict):
    mol_g_l = [local_molgraph_dict[c] for c in ik_list]
    bat_mol = dgl.batch(mol_g_l)
    return bat_mol

def get_single_cand_list(batch_size, cand_aug_random, qmol, cand_dict):
    if cand_aug_random:
        single_cand_l = random.choices(cand_dict[qmol][1], k=batch_size)
    else:
        start_cnt = cand_dict[qmol][0]
        totl = len(cand_dict[qmol][1])
        end_cnt = start_cnt + batch_size
        if totl < batch_size:
            single_cand_l = cand_dict[qmol][1] * int(batch_size/totl) + cand_dict[qmol][1][0:batch_size%totl]
        else:
            if end_cnt <= totl:
                single_cand_l = cand_dict[qmol][1][start_cnt:end_cnt]
                cand_dict[qmol][0] = end_cnt % totl
            else:
                rem_cnt = batch_size - (totl-start_cnt)
                single_cand_l = cand_dict[qmol][1][start_cnt:] + cand_dict[qmol][1][0:rem_cnt]
                cand_dict[qmol][0] = rem_cnt
                
    return single_cand_l
            
def get_cand_lists(params, molgraph_dict, cand_molgraph_dict, query_l, cand_dict, mol_dict, device):
    batch_size = params['batch_size_train_contr_cand']
    cand_aug_random = params['cand_aug_random']
    cand_l = [get_single_cand_list(batch_size, cand_aug_random, mol, cand_dict) for mol in query_l]
    cand_g_l = [[single_molgraph_return(c[0], molgraph_dict, cand_molgraph_dict, mol_dict, params, device) for c in cand] for cand in cand_l]
    bat_cand_g = [dgl.batch(cand_g) for cand_g in cand_g_l]
    bat_cand_g = dgl.batch(bat_cand_g)
    sim_l = [torch.tensor([0.0 if c[1] == 1.0 else c[1] for c in cand]) for cand in cand_l] #hack - make same molecule as 0.0
    
    return bat_cand_g, sim_l

class collate_contr_views(object):
    # class to hold and batch graph objects
    def __init__(self, *param_l):
        self.molgraph_dict = param_l[0]
        self.params = param_l[1]
        self.cand_dict_train = param_l[2]
        self.cand_molgraph_dict = param_l[3]
        self.mol_dict = param_l[4]
        self.device = param_l[5]
        self.cand_aug_random = self.params['cand_aug_random']
        self.mz_log_lims = (self.params['mz_log_low'], self.params['mz_log_high'])
        self.mz_spacing = self.params['mz_spacing']
        self.mz_precision = self.params['mz_precision']
        self.embd_dim = self.params['sinus_embed_dim']
        self.resolution = self.params['resolution']
        self.max_mz = self.params['max_mz']
        self.transformation = 'log10over3'
        if self.mz_precision is None:
            self.local_dtype = torch.float
        elif self.mz_precision == 16:
            self.local_dtype = torch.float16
        elif self.mz_precision == 32:
            self.local_dtype = torch.float32
        elif self.mz_precision == 64:
            self.local_dtype = torch.float64
        else:
            raise ValueError()
        self.spec_enc = self.params['spec_enc']
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        
    def __call__(self, batch):
        mz_l = [b[0][1][:,1] for b in batch]
        int_l = [b[0][1][:,0] for b in batch]
        if self.spec_enc == 'MLP_BIN':
            mzi = [b[0][1] for b in batch]
            mz_b, int_b, pad, lengths = get_ms_array_batch(mzi, self.transformation, self.max_mz, self.resolution)
        else:
            mz_b, int_b, pad, lengths = self.collate_mzi(mz_l, int_l)
        fp_l = [b[0][0][1] for b in batch]
        fp_b = np.vstack(fp_l)
        fp_b = torch.from_numpy(fp_b).to(dtype=torch.float32)
        batched_g = [self.molgraph_dict[b[1]] for b in batch]
        batched_g = dgl.batch(batched_g)
        if self.params['aug_cands']:
            query_l = [b[1] for b in batch]
            bat_cand_g, sim_l = get_cand_lists(self.params, self.molgraph_dict, self.cand_molgraph_dict, query_l, 
                                               self.cand_dict_train, self.mol_dict, self.device)
        else:
            bat_cand_g, sim_l = None, None
        ret_tuple = (batched_g, mz_b, int_b, pad, fp_b, lengths, bat_cand_g, sim_l)
        return ret_tuple
    
    
    def get_sinus_repr(self, mz_tensor):
        
        ####MODIFICATION MADE HERE: MATCH EQUATION ############
        if self.mz_spacing == 'log':
            frequency = 2 * np.pi / (10**self.mz_log_lims[0]*torch.logspace(
                0, 1, int(self.embd_dim / 2), base=(10**self.mz_log_lims[1])/(10**self.mz_log_lims[0]), dtype=self.local_dtype))
    
            #print check this is correct
            #print(frequency)
            #for D = 4 should be
            # [2pi/10e-2*(10e5)^(0/2), 2pi/10e-2*(10e5)^(2/2)]
            # [2pi/10e-2, 2pi/10e3]
            # [62.831, 0.00062]
    
        #######################################################
        elif self.mz_spacing == 'linear':
            frequency = 2 * np.pi / torch.linspace(
                self.mz_log_lims[0], self.mz_log_lims[1], int(self.embd_dim / 2), dtype=self.local_dtype)
        else:
            raise ValueError('mz_spacing must be either log or linear')
    
        omega_mz = frequency.reshape((1, 1, -1)) * mz_tensor
        sin = torch.sin(omega_mz)
        cos = torch.cos(omega_mz)
    
        ######## MODIFICATION MADE HERE: sin and cos will alternate ##
        #mz_vecs = torch.cat([sin, cos], axis=2)
    
        sin = sin.unsqueeze(3)
        cos = cos.unsqueeze(3)
        mz_vecs = torch.cat([sin, cos], axis=3)
        mz_vecs = mz_vecs.view(mz_vecs.shape[0], mz_vecs.shape[1], 2*sin.shape[2])
    
        return mz_vecs
        
    def collate_mzi(self, mz_l, int_l):
        lengths = np.array([len(mz) for mz in mz_l])
        b, l = len(mz_l), max(lengths)
        #tmp_tensor = torch.tensor((), dtype=self.local_dtype)
        mz_block = np.zeros((b, l+2))
    
        for i in range(b):
            #mz_block[i, 1:len(mz_l[i])-1] = mz_l[i][1:-1]
            mz_block[i, 1:len(mz_l[i])+1] = mz_l[i]
            if self.spec_enc == 'TFM':
                mz_block[i, 0] = self.SOS
                mz_block[i, lengths[i]+1] = self.EOS
    
        mz_b = torch.from_numpy(mz_block)
        mz_b = mz_b.to(self.local_dtype)
    
        lengths = np.array([len(inte) for inte in int_l])
        b, l = len(int_l), max(lengths)
        #tmp_tensor = torch.tensor((), dtype=self.local_dtype)
    
        int_block = np.zeros((b, l+2))
    
        for i in range(b):
            int_block[i, 1:len(int_l[i])+1] = int_l[i]
            if self.spec_enc == 'TFM':
                int_block[i, 0] = self.SOS
                int_block[i, lengths[i]+1] = self.EOS
    
        int_b = torch.from_numpy(int_block)
        int_b = int_b.to(self.local_dtype)
        int_b = int_b.unsqueeze(2)
        pad = mz_b == 0
        
        mz_b = mz_b.unsqueeze(2)
        mz_b = self.get_sinus_repr(mz_b)
        
        return mz_b, int_b, pad, lengths
    
class DatasetBuilder(object):
    """
    Loads the necessary files and creates the corresponding Dataset objects that are needed from this Builder class
    """
    
    def __init__(self, exp):
        """
        Loads the data correspoding to the experiment for construction of the Dataset objects later. Preprocess tokenized peaks and tokenized fingerprints for data_dict
        
        Parameters:
        -----------
        exp : str
            Experiment name of the dataset to load
        dir_path : str
            Path to where all the main files are located (typically just ./). Root directory containing data folder
        fp_path : str
            Path from data folder to the file containing the fingerprint (include fingerprint file name, EXCLUDE: extension)
        ms_intensity_threshold : int
            Minumum allowable intensity for a peak to be considered by the model
        tokenizer : Tokenizer
            Tokenizer object to parse spectra data into sequence
        old_sterr : stderr
            stderr file descriptor to write error messages into
        save_file : Boolean
            Specifies cache of data dict should be saved to store pre-processed data_dict
        load_file : Boolean
            Specifies cache of data_dict should be loaded to skip pre-processing of data_dict
        pos_mode : str
            Ionization mode so that the data will only load spectra of a specified ionization mode (See class SpectraDataset)

        Exception:
        ----------
        invalid_experiment : Exception
            Exception that specifies that the exp parameter was not a valid experiment type (see _init_exp for types)

        Fields:
        -------
        data_dict : dict
            dictionary of ms_id to spectra data. Augmented to include tokenized peaks and tokenized fingerprints
        mol_dict : dict
            dictionary of rdkit molecule objects for each molecule
        pair_dict : dict
            dictionary of translation ms_id pairs for each molecule
        split_dict : dict
            dictionary of the list of inchikeys in each data split
        fp_dict : dict
            dictionary of the np sparse data of each fingerprint of each inchikey
        """
        super(DatasetBuilder, self).__init__()

        
        self.exp = exp

    def init(self, dir_path, fp_path,ms_intensity_threshold):

        if self._init_exp() == False:
            raise Exception("Dataset: %s, is not available"%self.exp)

        data_dict, mol_dict, pair_dict, split_dict, fp_dict, in_to_id_dict, in_to_id_dict_wneg = self._load_dicts(dir_path, fp_path)

        if "test" in split_dict:
            self.test_inchis = split_dict['test']
        else:
            self.test_inchis = None
        


        #self.max_seq_len = max_seq_len


        self.data_dict = data_dict
        self.mol_dict = mol_dict
        self.pair_dict = pair_dict
        self.split_dict = split_dict
        self.fp_dict = fp_dict
        self.in_to_id_dict = in_to_id_dict
        self.in_to_id_dict_wneg = in_to_id_dict_wneg
        
        

    def _init_exp(self):
        """
        Internal function to initialize data for each experiment

        Returns:
        --------
        successful : bool
            Specifies that the experiment field (self.exp) is valid to initialize data from

        Field Accessor:
        ---------------
        exp : str
            string for the experiment to get dataset information on
        
        Field Mutator:
        --------------
        data_dir : str
            directory from project root to the folder containing the dataset data
        train_sample_num : int
            number of molecules to use for training (only used when validation is automatically split out of the train_split, not when a seperate valid_split is already specified)
        num_samples : int
            number of samples to use for Data Loader with random sampling
        """

        if self.exp == "canopus":
            data_dir = './data/NPLIB1/'
            self.train_sample_num = None
            self.num_samples = 100

            self.data_dir = data_dir
        else:
            return False

        return True

    def _load_dicts(self, dir_path, fp_path):
        """
        Internal function to obtain data of the specified dataset

        Parameters:
        -----------
        dir_path : str
            Path to where all the main files are located (typically just ./). Root directory containing data folder
        fp_path : str
            Path from data folder to the file containging the fingerprint (include fingerprint file name, EXCLUDE: extension)
        
        Returns:
        --------
        data_dict : dict
            dictionary of ms_id to spectra data. Augmented to include tokenized peaks and tokenized fingerprints
        mol_dict : dict
            dictionary of rdkit molecule objects for each molecule
        pair_dict : dict
            dictionary of translation ms_id pairs for each molecule
        split_dict : dict
            dictionary of the list of inchikeys in each data split
        fp_dict : dict
            dictionary of the np sparse data of each fingerprint of each inchikey
        """

        data_dict = self._load_dict(dir_path + self.data_dir + "data_dict.pkl")
        mol_dict = self._load_dict(dir_path + self.data_dir + 'mol_dict.pkl')
        pair_dict = None #self._load_dict(dir_path + self.data_dir + 'pair_permutation_ids.pkl')
        split_dict = self._load_dict(dir_path + self.data_dir + 'split.pkl')
        #fp_dict = self._load_dict(dir_path + self.data_dir + fp_path + '.pkl')
        fp_dict = None
        in_to_id_dict = self._load_dict(dir_path + self.data_dir + 'inchi_to_id_dict.pkl')
        in_to_id_dict_wneg = self._load_dict(dir_path + self.data_dir + 'inchi_to_id_dict_wneg.pkl')
        
        return data_dict, mol_dict, pair_dict, split_dict, fp_dict, in_to_id_dict, in_to_id_dict_wneg

    def _load_dict(self, path):
        """
        Internal function to get pickle data from the specified file path

        Parameters:
        -----------
        path : str
            Path to pickle file to load

        Returns:
        --------
        results_dict : dict
            Returns the dict stored in the pickle file           
        """

        with open(path, 'rb') as f:
            results_dict = pickle.load(f)

        return results_dict


    def make_spectra_data(self):
        """
        Creates and internally stores the SpectraDataset objects for each split. Expects that dicts are still available (before build() function call)

        Mutator:
        --------
        spec_train : SpectraDataset
            training spectra from split_dict inchis
        spec_valid : SpectraDataset
            validation spectra from split_dict inchis
        spec_test : SpectraDataset
            testing spectra from split_dict inchis

        Returns:
        --------
        self : DatasetBuilder
            returns own instance for easier stacking interface
        """

        my_dataset = SpectraDataset
        
        if self.train_sample_num is None:
            train_data = my_dataset(self.split_dict['train'], self.pair_dict, self.data_dict, self.mol_dict, self.tokenizer, charge_mode=self.pos_mode)
            val_data = my_dataset(self.split_dict['valid'], self.pair_dict, self.data_dict, self.mol_dict, self.tokenizer, charge_mode=self.pos_mode)
            test_data = my_dataset(self.split_dict['test'], self.pair_dict, self.data_dict, self.mol_dict, self.tokenizer, charge_mode=self.pos_mode)

        else:
            train_data = my_dataset(self.split_dict['train'][:self.train_sample_num], self.pair_dict, self.data_dict, self.mol_dict, self.tokenizer, charge_mode=self.pos_mode)
            val_data = my_dataset(self.split_dict['train'][self.train_sample_num:], self.pair_dict, self.data_dict, self.mol_dict, self.tokenizer, charge_mode=self.pos_mode)
            test_data = my_dataset(self.split_dict['test'], self.pair_dict, self.data_dict, self.mol_dict, self.tokenizer, charge_mode=self.pos_mode)

        self.spec_train = train_data
        self.spec_valid = val_data
        self.spec_test = test_data

        return self


    def build(self):
        """
        Function to call after createing all data to finalize building before accessing the created data. Destroys refernces to all dict objects to save memory

        Destroys Fields:
        ----------------
        data_dict : dict
            dictionary of ms_id to spectra data. Augmented to include tokenized peaks and tokenized fingerprints
        mol_dict : dict
            dictionary of rdkit molecule objects for each molecule
        pair_dict : dict
            dictionary of translation ms_id pairs for each molecule
        split_dict : dict
            dictionary of the list of inchikeys in each data split
        fp_dict : dict
            dictionary of the np sparse data of each fingerprint of each inchikey
        """

        self.data_dict = None
        self.mol_dict = None
        self.pair_dict = None
        self.split_dict = None
        self.fp_dict = None

    def get_spectra_data(self):
        """
        Function returns SpectraDataset objects for all splits

        Returns:
        --------
        spec_train : SpectraDataset
            training spectra from split_dict inchis
        spec_valid : SpectraDataset
            validation spectra from split_dict inchis
        spec_test : SpectraDataset
            testing spectra from split_dict inchis
        """

        return self.spec_train, self.spec_valid, self.spec_test


    
    def get_max_seq_len(self):
        return self.max_seq_len

    def get_num_samples(self):
        return self.num_samples

    def get_test_inchis(self):
        return self.test_inchis

def contrastive_loss(v1, v2, tau=1.0):
    
    v1_norm = torch.norm(v1, dim=1, keepdim=True)
    v2_norm = torch.norm(v2, dim=1, keepdim=True)
    
    v2T = torch.transpose(v2, 0, 1)
    
    inner_prod = torch.matmul(v1, v2T)
    
    v2_normT = torch.transpose(v2_norm, 0, 1)
    
    norm_mat = torch.matmul(v1_norm, v2_normT)
    
    loss_mat = torch.div(inner_prod, norm_mat)
    
    loss_mat = loss_mat * (1/tau)
    
    loss_mat = torch.exp(loss_mat)
    
    numerator = torch.diagonal(loss_mat)
    numerator = torch.unsqueeze(numerator, 0)
    
    Lv1_v2_denom = torch.sum(loss_mat, dim=1, keepdim=True)
    Lv1_v2_denom = torch.transpose(Lv1_v2_denom, 0, 1)
    #Lv1_v2_denom = Lv1_v2_denom - numerator
    
    Lv2_v1_denom = torch.sum(loss_mat, dim=0, keepdim=True)
    #Lv2_v1_denom = Lv2_v1_denom - numerator
    
    Lv1_v2 = torch.div(numerator, Lv1_v2_denom)
    
    Lv1_v2 = -1 * torch.log(Lv1_v2)
    Lv1_v2 = torch.mean(Lv1_v2)
    
    Lv2_v1 = torch.div(numerator, Lv2_v1_denom)
    
    Lv2_v1 = -1 * torch.log(Lv2_v1)
    Lv2_v1 = torch.mean(Lv2_v1)
    
    return Lv1_v2 + Lv2_v1

def augmented_cand_loss(mol_enc, cand_mol_enc_l, sim_l):
    cand_enc = torch.stack(cand_mol_enc_l, dim = 2)
    mol_enc = torch.transpose(mol_enc, 0, 1)
    mol_enc = mol_enc.unsqueeze(0)
    sim = torch.stack(sim_l, dim = 1)
    #sim = torch.squeeze(1)
    cos = torch.nn.CosineSimilarity()
    cos_dist = cos(mol_enc, cand_enc)
    loss = cos_dist - sim
    loss = torch.sum(loss)
    
    return loss

#def augmented_cand_loss_spec(spec_enc, cand_mol_enc_l, sim_l):
def augmented_cand_loss_spec(spec_enc, cand_enc, sim_l):
    #cand_enc = torch.stack(cand_mol_enc_l, dim = 2)
    #spec_enc = torch.transpose(spec_enc, 0, 1)
    cand_enc = torch.transpose(cand_enc, 0, 1)
    spec_enc = spec_enc.unsqueeze(0)
    sim = torch.stack(sim_l, dim = 1)
    #sim = torch.squeeze(1)
    cos = torch.nn.CosineSimilarity(dim=2)
    cos_dist = cos(spec_enc, cand_enc)
    loss = torch.mean(cos_dist)
    
    return loss

def fp_bce_loss(fp_pred, fp_gt):
    weight = fp_gt * 0
    weight[fp_gt >= 0.5] = (fp_gt.shape[0]*fp_gt.shape[1])/(torch.sum(fp_gt >= 0.5)*2)
    weight[fp_gt < 0.5] = (fp_gt.shape[0]*fp_gt.shape[1])/(torch.sum(fp_gt < 0.5)*2)
    bce_loss = torch.nn.BCELoss()
    #bce_loss = torch.nn.BCELoss(weight)
    fp_loss = bce_loss(fp_pred, fp_gt)

    #mse_loss = torch.nn.MSELoss(reduction='none')
    #fp_loss = torch.mean(weight*mse_loss(fp_pred, fp_gt))-1
        

    return fp_loss

def fp_cos(fp_pred, fp_gt):
    cos = torch.nn.CosineSimilarity()
    cosr = cos(fp_pred, fp_gt)
    cosr = torch.mean(cosr)
    return cosr

def fp_cos_loss(fp_pred, fp_gt):
    cos = torch.nn.CosineSimilarity()
    cos_loss = 1 - cos(fp_pred, fp_gt)
    cos_loss = torch.mean(cos_loss)
    return cos_loss

def load_models(params, models, device, output):
    """ load models if pretrained_models are available """
    for m in range(len(models)):
        model, idx = models[m][0], models[m][1]
        idx = "pretrained_enz_model" if idx == "" else "pretrained_%s_model" % idx
        idx_path = params.get(idx)
        if idx_path is not None:
            Print('loading %s weights from %s' % (idx, params[idx]), output)
            models[m][0].load_weights(params[idx])

        models[m][0] = models[m][0].to(device)
        
def Print(string, output, newline=False):
    """ print to stdout and a file (if given) """
    time = datetime.now()
    print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=sys.stderr)
    if newline: print("", file=sys.stderr)

    if not output == sys.stdout:
        print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=output)
        if newline: print("", file=output)

    output.flush()
    return time

def print_hp(hp, output):
    hplist = []
    hplist.append("Values of Hyperparameters:")
    for key, value in hp.items():
        hplist.append("{} : {}".format(key, value))
    for str in hplist:
        Print(str, output)
        
class MyEarlyStopping(EarlyStopping):
    def __init__(self, models_list, model_time, hp, output, mode='higher', patience=10, filename=None, metric=None):
        super(MyEarlyStopping, self).__init__(mode=mode, patience=patience, filename=filename, metric=metric)
        self.models_list = models_list
        self.model_time = model_time
        self.hp = hp
        self.output = output

    def step(self, score):
        """Update based on a new score.

        The new score is typically model performance on the validation set
        for a new epoch.

        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.

        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            save_all_models(self.hp, self.models_list, self.model_time + "_best", self.output)
        elif self._check(score, self.best_score):
            self.best_score = score
            save_all_models(self.hp, self.models_list, self.model_time + "_best", self.output)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            save_all_models(self.hp, self.models_list, self.model_time + "_current", self.output)

        return self.early_stop
    
def load_models(hp, models, device, output):
    """ load models if pretrained_models are available """
    for m in range(len(models)):
        model, idx = models[m][0], models[m][1]
        idx = "pretrained_enz_model" if idx == "" else "pretrained_%s_model" % idx
        idx_path = hp.get(idx)
        if idx_path is not None:
            Print('loading %s weights from %s' % (idx, hp[idx]), output)
            models[m][0].load_weights(hp[idx])

        models[m][0] = models[m][0].to(device)

def save_all_models(hp, models, model_time, output):
    for m in range(len(models)):
        model, idx = models[m][0], models[m][1]
        idx = "pretrained_enz_model" if idx == "" else "pretrained_%s_model" % idx
        filename = hp['data_dir'] + idx + '_' + model_time + '.pt'
        save_model(model, filename, output)
        
def save_model(model, filename, output):
    Print('saving %s weights to %s' % (model.__class__.__name__, filename), output)
    torch.save(model.state_dict(), filename)
    
def set_saved_best_model_names(hp, models, model_time):
    for m in range(len(models)):
        model, idx = models[m][0], models[m][1]
        idx = "pretrained_enz_model" if idx == "" else "pretrained_%s_model" % idx
        filename = hp['data_dir'] + idx + '_' + model_time + '_best.pt'
        hp[idx] = filename

def set_seeds(seed):
    """ set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
