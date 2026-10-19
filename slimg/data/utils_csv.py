import torch
import numpy as np
from torch_geometric.data import Data


def load_csv(fname_edges: str, fname_features: str, fname_labels: str) -> Data:
    edges = np.loadtxt(fname_edges, delimiter=",", dtype=np.int64)
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    x = torch.tensor(np.loadtxt(fname_features, delimiter=",", dtype=np.float32))
    data = Data(edge_index=edge_index, x=x)

    y = torch.tensor(np.loadtxt(fname_labels, delimiter=",", dtype=np.int64))
    data.y = y

    return data
