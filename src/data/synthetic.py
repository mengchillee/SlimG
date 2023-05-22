import math

import torch
from torch import spmm
from torch_geometric.data import InMemoryDataset, Data

from models.utils import normalize_adj

CASES = [
    ('random', 'clustered', 'homophily'),
    ('random', 'bipartite', 'heterophily'),
    ('semantic', 'uniform', 'individual'),
    ('semantic', 'clustered', 'individual'),
    ('semantic', 'bipartite', 'individual'),
    ('structural', 'clustered', 'homophily'),
    ('structural', 'bipartite', 'heterophily'),
    ('semantic', 'clustered', 'homophily'),
    ('semantic', 'bipartite', 'heterophily'),
]


def to_adj_matrix(edge_index, num_nodes):
    return torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)),
                                   (num_nodes, num_nodes))


def normalize_features(x):
    std = x.std(0)
    out = x - x.mean(0)
    out[:, std > 0] /= std[std > 0]
    return out


def make_intra_edges(nodes, density):
    row, col = [], []
    for i, n in enumerate(nodes):
        dst_idx, = torch.nonzero(torch.rand(len(nodes) - i - 1) < density,
                                 as_tuple=True)
        row.append(torch.full((len(dst_idx),), fill_value=n))
        col.append(nodes[dst_idx + i + 1])
    row = torch.cat(row)
    col = torch.cat(col)
    row_out = torch.cat([row, col])
    col_out = torch.cat([col, row])
    return torch.stack([row_out, col_out])


def make_inter_edges(nodes1, nodes2, density):
    row, col = [], []
    for n in nodes1:
        dst_idx, = torch.nonzero(torch.rand(len(nodes2)) < density,
                                 as_tuple=True)
        row.append(torch.full((len(dst_idx),), fill_value=n))
        col.append(nodes2[dst_idx])
    row = torch.cat(row)
    col = torch.cat(col)
    row_out = torch.cat([row, col])
    col_out = torch.cat([col, row])
    return torch.stack([row_out, col_out])


def make_clustered_edges(clusters, pairs, density, noise):
    def select_nodes(c):
        return torch.nonzero(clusters == c, as_tuple=True)[0]

    edge_list = []
    num_clusters = (clusters.max() + 1).item()
    for c1 in range(num_clusters):
        for c2 in range(c1, num_clusters):
            if (c1, c2) in pairs:
                d = density
            else:
                d = noise * density
            if c1 == c2:
                edges = make_intra_edges(select_nodes(c1), d)
            else:
                edges = make_inter_edges(select_nodes(c1), select_nodes(c2), d)
            edge_list.append(edges)
    return torch.cat(edge_list, dim=1)


def assign_clusters(num_nodes, num_clusters):
    clusters = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_clusters):
        clusters[int(i * num_nodes / num_clusters):
                 int((i + 1) * num_nodes / num_clusters)] = i
    return clusters


def propagate_features(init_features, edge_index, num_nodes, num_steps=2):
    adj = normalize_adj(edge_index, num_nodes, direction='row')
    out = init_features
    for _ in range(num_steps):
        out = spmm(adj, out)
    return out


def assign_balanced_labels(scores):
    num_nodes = scores.size(0)
    num_classes = scores.size(1)

    node_indices = torch.arange(num_nodes).repeat_interleave(num_classes)
    label_indices = torch.arange(num_classes).repeat(num_nodes)
    score_flattened = scores.view(-1)

    count_limit = math.ceil(num_nodes / num_classes)
    label_counts = torch.zeros(num_classes, dtype=torch.int64)
    labels = torch.full((num_nodes,), fill_value=-1, dtype=torch.int64)

    for i in torch.argsort(score_flattened, descending=True):
        node = node_indices[i]
        label = label_indices[i]
        if labels[node] == -1 and label_counts[label] < count_limit:
            labels[node] = label
            label_counts[label] += 1
    return labels


class Synthetic(InMemoryDataset):
    def __init__(self, num_nodes=6000, num_features=500, num_classes=4,
                 x_type='semantic', e_type='clustered', y_type='homophily',
                 edge_density=0.01, edge_noise=0.4, feature_noise=0.):
        super().__init__()
        torch.manual_seed(0)

        self.num_nodes = num_nodes
        self.num_classes_ = num_classes
        self.edge_density = edge_density
        self.edge_noise = edge_noise
        self.feature_noise = feature_noise

        self.e_type = e_type
        self.y_type = y_type
        self.x_type = x_type

        if num_features is None:
            num_features = num_classes * (num_classes + 1)

        edge_index = self.make_adjacency()
        y = self.make_labels()
        x = self.make_features(edge_index, y, num_features)

        self.data = Data(x, edge_index, y=y)

    @property
    def num_classes(self):
        return self.num_classes_

    def pick_class_pairs(self):
        assert self.num_classes % 2 == 0
        return [(2 * i, 2 * i + 1) for i in range(self.num_classes // 2)]

    def make_adjacency(self):
        if self.e_type == 'uniform':
            nodes = torch.arange(self.num_nodes)
            c = self.num_classes
            d = self.edge_density
            e = self.edge_noise
            density = (d * (1 + (c - 1) * e)) / c
            return make_intra_edges(nodes, density)
        elif self.e_type in ['clustered', 'bipartite']:
            if self.e_type == 'clustered':
                pairs = [(c, c) for c in range(self.num_classes)]
            else:
                pairs = self.pick_class_pairs()
            clusters = assign_clusters(self.num_nodes, self.num_classes)
            return make_clustered_edges(clusters, pairs, self.edge_density,
                                        self.edge_noise)
        else:
            raise ValueError(self.e_type)

    def make_labels(self):
        if self.y_type == 'individual':
            return torch.randint(self.num_classes, (self.num_nodes,))
        elif self.e_type == 'clustered' and self.y_type == 'homophily':
            return assign_clusters(self.num_nodes, self.num_classes)
        elif self.e_type == 'bipartite' and self.y_type == 'heterophily':
            return assign_clusters(self.num_nodes, self.num_classes)
        else:
            raise ValueError(self.e_type, self.y_type)

    def make_features(self, edge_index, labels, num_features):
        def make_semantic_features():
            centers = torch.rand((num_features, self.num_classes))
            centers /= centers.sum(0)
            out_ = torch.zeros((self.num_nodes, num_features))
            mask_ = torch.zeros(self.num_nodes, dtype=torch.bool)
            while not torch.all(mask_):
                random = torch.rand((self.num_nodes, num_features))
                out_labels = random.mm(centers).argmax(1)
                index = (out_labels == labels) & ~mask_
                out_[index] = random[index]
                mask_[index] = True
            return out_

        def make_structural_features():
            adj = to_adj_matrix(edge_index, self.num_nodes)
            u, s, v = torch.svd_lowrank(adj, num_features)
            return normalize_features(torch.relu(u) * s.unsqueeze(0))

        if self.x_type == 'random':
            out = torch.rand((self.num_nodes, num_features))
        elif self.x_type == 'semantic':
            out = make_semantic_features()
        elif self.x_type == 'structural' and self.e_type != 'uniform':
            out = make_structural_features()
        else:
            raise ValueError(self.x_type)

        if self.feature_noise > 0:
            mask = torch.rand_like(out) < self.feature_noise
            out[mask] = torch.randn_like(out)[mask]
        return out
