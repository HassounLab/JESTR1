# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:18:01 2023

@author: apurv
"""
import torch
import torch.nn as nn
import dgl
from dgllife.model import GCN, GAT

class MolEnc(nn.Module):
    def __init__(self, params, in_dim):
        super(MolEnc, self).__init__()
        dropout = [params['gnn_dropout'] for _ in range(len(params['gnn_channels']))]
        batchnorm = [True for _ in range(len(params['gnn_channels']))]
        gnn_map = {
            "gcn": GCN(in_dim, params['gnn_channels'], batchnorm = batchnorm, dropout = dropout),
            "gat": GAT(in_dim, params['gnn_channels'], params['attn_heads'])
        }
        self.GNN = gnn_map[params['gnn_type']]
        self.pool = dgl.nn.pytorch.glob.MaxPooling()
        self.fc1_graph = nn.Linear(params['gnn_channels'][len(params['gnn_channels']) - 1], params['gnn_hidden_dim'] * 2)
        self.fc2_graph = nn.Linear(params['gnn_hidden_dim'] * 2, params['final_embedding_dim'])
        self.W_out1 = nn.Linear(params['final_embedding_dim'], params['gnn_hidden_dim'] * 2)
        self.W_out2 = nn.Linear(params['gnn_hidden_dim'] * 2, params['gnn_hidden_dim'])
        self.dropout = nn.Dropout(params['fc_dropout'])
        self.relu = nn.ReLU()

    def forward(self, g1, f1):
        f = self.GNN(g1, f1)
        h = self.pool(g1, f)
        h1 = self.relu(self.fc1_graph(h))
        h1 = self.dropout(h1)
        h1 = self.fc2_graph(h1)
        h1 = self.dropout(h1)
        #h1 = self.relu(h1)
        
        return h1

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)
    
class SpecEncMLP_BIN(nn.Module):
    def __init__(self, params, bin_size):
        super(SpecEncMLP_BIN, self).__init__()

        self.dropout = nn.Dropout(params['fc_dropout'])
        self.mz_fc1 = nn.Linear(bin_size, params['final_embedding_dim'] * 2)
        self.mz_fc2 = nn.Linear(params['final_embedding_dim'] * 2, params['final_embedding_dim'] * 2)
        self.mz_fc3 = nn.Linear(params['final_embedding_dim'] * 2, params['final_embedding_dim'])
        self.relu = nn.ReLU()
        self.aggr_method = params['aggregator']
        
    def aggr(self, mzvec):
        if self.aggr_method == 'sum':
            aggr_ret = torch.sum(mzvec, axis=1)
        elif self.aggr_method == 'mean':
            aggr_ret = mzvec.sum(axis=1) / ~pad.sum(axis=-1).unsqueeze(-1)
        elif self.aggr_method == 'maxpool':
            input_mask_expanded = torch.where(pad==True, -1e-9, 0.).unsqueeze(-1).expand(mzvec.size()).float()
            aggr_ret = torch.max(mzvec-input_mask_expanded, 1)[0] # Set padding tokens to large negative value
            
        return aggr_ret
    
    def forward(self, mzi_b, int_b, pad, lengths):
                
       h1 = self.mz_fc1(mzi_b)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       h1 = self.mz_fc2(h1)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       mz_vec = self.mz_fc3(h1)
       mz_vec = self.dropout(mz_vec)
              
       #mz_vec = self.aggr(mz_vec)
       
       return mz_vec

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)

        
class INTER_MLP2(nn.Module):
    def __init__(self, params):
        super(INTER_MLP2, self).__init__()
        self.dropout = nn.Dropout(params['fc_dropout'])
        self.fp_fc1 = nn.Linear(params['final_embedding_dim'] * 2, params['final_embedding_dim'])
        self.fp_fc2 = nn.Linear(params['final_embedding_dim'], params['final_embedding_dim'] // 2)
        self.fp_fc3 = nn.Linear(params['final_embedding_dim'] // 2, params['final_embedding_dim'] // 4)
        self.fp_fc4 = nn.Linear(params['final_embedding_dim'] // 4, params['final_embedding_dim'] // 8)
        self.fp_fc5 = nn.Linear(params['final_embedding_dim'] // 8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, mol, spec):
        
        y_cat = torch.cat((mol, spec), 1) 
            
        h = self.fp_fc1(y_cat)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc3(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc4(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        h = self.fp_fc5(h)
        z_interaction = self.sigmoid(h)
       
        return z_interaction

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)
