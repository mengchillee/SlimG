import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.decomposition import PCA as PCA_CPU
from models.pca import PCA as PCA_GPU

from models.utils import normalize_adj


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


class OurModel(nn.Module):
    def __init__(self, num_features, num_nodes, num_classes, pca_gpu=False):
        super().__init__()
        self.num_classes = num_classes
        self.pca_gpu = pca_gpu
        self.struct_x = None
        self.x = None
        self.l_x = []

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for l in self.linears:
            l.reset_parameters()
        return self

    def preprocess(self, x, edge_index, labels, device):
        self.struct_x = make_structural_features(edge_index, x.size(0), np.min([*x.shape]), device)
        self.struct_x = F.normalize(self.struct_x, p=2, dim=1)

        if not self.pca_gpu:
            self.x = torch.tensor(PCA_CPU(n_components=self.struct_x.shape[1]).fit_transform(x.cpu().numpy())).float().to(device)
        else:
            self.x = PCA_GPU(n_components=self.struct_x.shape[1]).fit_transform(x)
        self.x = F.normalize(self.x, p=2, dim=1)

        f = propagate(x, edge_index, 2, 'row', self_loops=False)
        if not self.pca_gpu:
            f = torch.tensor(PCA_CPU(n_components=self.x.shape[1]).fit_transform(f.cpu().numpy())).float().to(device)
        else:
            f = PCA_GPU(n_components=self.x.shape[1]).fit_transform(f)
        f = F.normalize(f, p=2, dim=1)
        self.l_x.append(f)

        f = propagate(x, edge_index, 2, 'sym', self_loops=True)
        if not self.pca_gpu:
            f = torch.tensor(PCA_CPU(n_components=self.x.shape[1]).fit_transform(f.cpu().numpy())).float().to(device)
        else:
            f = PCA_GPU(n_components=self.x.shape[1]).fit_transform(f)
        f = F.normalize(f, p=2, dim=1)
        self.l_x.append(f)

        self.lin1 = nn.Linear(self.struct_x.shape[1], self.num_classes, bias=True).to(device)
        self.lin2 = nn.Linear(self.x.shape[1], self.num_classes, bias=True).to(device)
        linears = []
        for i in range(len(self.l_x)):
            linears.append(nn.Linear(self.l_x[i].shape[1], self.num_classes, bias=True).to(device))
        self.linears = nn.ModuleList(linears)

    def feature_size(self):
        shapes = np.array([self.struct_x.shape[1], self.x.shape[1]] + [l.shape[1] for l in self.l_x])

    def forward(self, x, edge_index):
        out = [self.lin1(self.struct_x), self.lin2(self.x)]
        for i, l in enumerate(self.l_x):
            out.append(self.linears[i](l))
        return torch.stack(out).sum(0)
