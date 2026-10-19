import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.decomposition import PCA as PCA_CPU
from .pca import PCA as PCA_GPU

from .utils import normalize_adj


def to_adj_matrix(edge_index, num_nodes):
    return torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))


def make_structural_features(edge_index, num_nodes, num_features, device):
    adj = to_adj_matrix(edge_index.cpu(), num_nodes).to(device)

    f_norm = (torch.norm(adj) ** 2).cpu().numpy()
    u, s, _ = torch.svd_lowrank(adj, int(num_features))

    if np.sum(s.cpu().numpy() ** 2) / f_norm < 0.9:
        return u * s.unsqueeze(0)
    else:
        u, s, _ = torch.svd_lowrank(adj, int(adj.shape[0] * 0.8))
        sarr = s.cpu().numpy() ** 2
        idx = np.where(np.cumsum(sarr) / f_norm >= 0.9)[0][0]
        return u[:, :idx] * s[:idx].unsqueeze(0)


def propagate(x, edge_index, num_layers, direction, self_loops):
    adj = normalize_adj(edge_index, len(x), direction=direction, self_loops=self_loops)
    x = x.clone()
    for _ in range(num_layers):
        x = torch.spmm(adj, x)
    return x


ALL_COMPONENTS = ['C1', 'C2', 'C3', 'C4']


class SlimG(nn.Module):
    def __init__(self, num_features, num_nodes, num_classes, pca_gpu=False, components=None):
        super().__init__()
        self.num_classes = num_classes
        self.pca_gpu = pca_gpu
        self.components = components if components is not None else ALL_COMPONENTS

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        return self

    def preprocess(self, x, edge_index, labels, device):
        inputs = []

        if 'C1' in self.components:
            struct_x = make_structural_features(edge_index, x.size(0), np.min([*x.shape]), device)
            struct_x = F.normalize(struct_x, p=2, dim=1)
            inputs.append(struct_x)
            pca_dim = struct_x.shape[1]
        else:
            pca_dim = np.min([*x.shape])

        if 'C2' in self.components:
            if not self.pca_gpu:
                px = torch.tensor(PCA_CPU(n_components=pca_dim).fit_transform(x.cpu().numpy())).float().to(device)
            else:
                px = PCA_GPU(n_components=pca_dim).fit_transform(x)
            px = F.normalize(px, p=2, dim=1)
            inputs.append(px)
            prop_dim = px.shape[1]
        else:
            prop_dim = pca_dim

        if 'C3' in self.components:
            f = propagate(x, edge_index, 2, 'row', self_loops=False)
            if not self.pca_gpu:
                f = torch.tensor(PCA_CPU(n_components=prop_dim).fit_transform(f.cpu().numpy())).float().to(device)
            else:
                f = PCA_GPU(n_components=prop_dim).fit_transform(f)
            f = F.normalize(f, p=2, dim=1)
            inputs.append(f)

        if 'C4' in self.components:
            f = propagate(x, edge_index, 2, 'sym', self_loops=True)
            if not self.pca_gpu:
                f = torch.tensor(PCA_CPU(n_components=prop_dim).fit_transform(f.cpu().numpy())).float().to(device)
            else:
                f = PCA_GPU(n_components=prop_dim).fit_transform(f)
            f = F.normalize(f, p=2, dim=1)
            inputs.append(f)

        self.inputs = inputs
        linears = []
        for inp in self.inputs:
            linears.append(nn.Linear(inp.shape[1], self.num_classes, bias=True).to(device))
        self.linears = nn.ModuleList(linears)

    def forward(self, x, edge_index):
        out = []
        for inp, lin in zip(self.inputs, self.linears):
            out.append(lin(inp))
        return torch.stack(out).sum(0)
