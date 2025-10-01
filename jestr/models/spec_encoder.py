import torch.nn as nn
import torch
from jestr.models.encoders import MLP
from torch_geometric.nn import global_mean_pool


class SpecEncMLP_BIN(nn.Module):
    def __init__(self, args, out_dim=None):
        super(SpecEncMLP_BIN, self).__init__()

        if not out_dim:
            out_dim = args.final_embedding_dim

        bin_size = int(args.max_mz / args.bin_width)
        self.dropout = nn.Dropout(args.fc_dropout)
        self.mz_fc1 = nn.Linear(bin_size, out_dim * 2)
        self.mz_fc2 = nn.Linear(out_dim* 2, out_dim * 2)
        self.mz_fc3 = nn.Linear(out_dim * 2, out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, mzi_b, n_peaks=None):
                
       h1 = self.mz_fc1(mzi_b)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       h1 = self.mz_fc2(h1)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       mz_vec = self.mz_fc3(h1)
       mz_vec = self.dropout(mz_vec)
       
       return mz_vec
    

