import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.1, final_activation=None):
        super(MLP, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.has_final_activation = False
        layers = [nn.Linear(in_dim, hidden_dims[0])]
        for d1, d2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(d1, d2))
        self.layers = nn.ModuleList(layers)
        if final_activation is not None:
            self.has_final_activation = True
    
            self.final_activation = {'relu': F.relu,
                                    'sigmoid': F.sigmoid,
                                    'softmax': F.softmax,}[final_activation]
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) -1:
                x = F.relu(x)
                x = self.dropout(x)
            elif self.has_final_activation:
                x = self.final_activation(x)
        return x
