import torch
import torch.nn as nn
import dgl
from dgllife.model import GCN, GAT

class MolEnc(nn.Module):

    def __init__(self,
                 args,
                 in_dim,):
        super().__init__()

        dropout = [args.gnn_dropout for _ in range(len(args.gnn_channels))]
        batchnorm = [True for _ in range(len(args.gnn_channels))]
        gnn_map = {
            "gcn": GCN(in_dim, args.gnn_channels, batchnorm = batchnorm, dropout = dropout),
            "gat": GAT(in_dim, args.gnn_channels, args.attn_heads)
        }
        self.GNN = gnn_map[args.gnn_type]
        self.pool = dgl.nn.pytorch.glob.MaxPooling()
    
        self.fc1_graph = nn.Linear(args.gnn_channels[len(args.gnn_channels) - 1], args.gnn_hidden_dim * 2)
        self.fc2_graph = nn.Linear(args.gnn_hidden_dim * 2, args.final_embedding_dim)

        self.dropout = nn.Dropout(args.fc_dropout)
        self.relu = nn.ReLU()

    def forward(self, g) -> torch.Tensor:
        g1 = g
        f1 = g.ndata['h']
        f = self.GNN(g1, f1)
        h = self.pool(g1, f)
        h1 = self.relu(self.fc1_graph(h))
        h1 = self.dropout(h1)
        h1 = self.fc2_graph(h1)
        h1 = self.dropout(h1)

        return h1
    
