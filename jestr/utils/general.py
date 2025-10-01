import torch
from torch import nn
import torch.nn.functional as F

def pad_graph_nodes(mol_enc, g_n_nodes):
    """
    Args:
        mol_enc: 2D tensor of shape (sum_nodes, D)
                 Node embeddings for each molecule.
        g_n_nodes: list[int]  Number of nodes per graph (len = B)

    Returns:
        padded: (B, max_nodes, D) tensor
        mask:   (B, max_nodes) bool tensor, True for valid nodes
    """

    # Already concatenated: shape (sum_nodes, D)
    B = len(g_n_nodes)
    D = mol_enc.shape[1]
    max_nodes = max(g_n_nodes)
    padded = mol_enc.new_zeros((B, max_nodes, D))
    mask = torch.zeros((B, max_nodes), dtype=torch.bool, device=mol_enc.device)

    idx = 0
    for i, n in enumerate(g_n_nodes):
        padded[i, :n] = mol_enc[idx:idx+n]
        mask[i, :n] = True
        idx += n
    return padded, mask