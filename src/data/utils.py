import numpy as np
import os
from os import path
import scipy
import gdown
import pandas as pd

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, Amazon, WebKB, LINKXDataset, SNAPDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.data import Data

from data.synthetic import Synthetic
from utils import ROOT


def print_data_stats(features, edge_index, labels):
    is_directed = not is_undirected(edge_index)
    if is_directed:
        num_edges = edge_index.size(1)
    else:
        num_edges = edge_index.size(1) // 2

    print(f'Directed: {is_directed}')
    print(f'Number of nodes: {features.size(0)}')
    print(f'Number of features: {features.size(1)}')
    print(f'Number of labels: {labels.max() + 1}')
    print(f'Number of edges: {num_edges}')
    print(f'Number of self-loops: {sum(edge_index[0] == edge_index[1])}')
    print(f'Min features: {features.min()}')
    print(f'Max features: {features.max()}')
    print(f'Min standard deviation: {features.std(dim=0).min()}')
    print(f'Max standard deviation: {features.std(dim=0).max()}')
    print()


def split_nodes(labels, ratio=(0.6, 0.2, 0.2), seed=0):
    assert len(ratio) == 3 and sum(ratio) == 1
    num_nodes = len(labels)
    trn_idx, test_idx = train_test_split(np.arange(num_nodes),
                                         test_size=ratio[2],
                                         stratify=labels,
                                         random_state=seed)
    trn_idx, val_idx = train_test_split(trn_idx,
                                        test_size=ratio[1] / (ratio[0] + ratio[1]),
                                        stratify=labels[trn_idx],
                                        random_state=seed)

    trn_mask = torch.zeros(num_nodes, dtype=torch.bool)
    trn_mask[trn_idx] = 1
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_idx] = 1
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = 1
    return trn_mask, val_mask, test_mask


def split_nodes_by_years(node_year, val_cut=2010, test_cut=2012):
    node_year = node_year.reshape(-1)
    trn_mask = torch.tensor(node_year < val_cut)
    val_mask = torch.tensor((node_year >= val_cut) & (node_year < test_cut))
    test_mask = torch.tensor(node_year >= test_cut)
    return trn_mask, val_mask, test_mask


def split_nodes_by_labels(labels, size=20):
    num_nodes = labels.size(0)
    num_labels = labels.max().item() + 1
    out = []
    for i in range(num_labels):
        nonzero = torch.nonzero(labels == i)
        selected = torch.randperm(len(nonzero))[:size]
        out.append(nonzero[selected])
    trn_nodes = torch.cat(out)
    trn_mask = torch.zeros(num_nodes, dtype=torch.bool)
    trn_mask[trn_nodes] = 1
    test_mask = ~trn_mask
    return trn_mask, test_mask


def load_data(name, root=f'{ROOT}/data', verbose=False):
    if not path.exists(root):
        os.makedirs(root)

    if name in ['arxiv', 'products']:
        graph = PygNodePropPredDataset('ogbn-' + name, root)
        graph.data.y = graph.data.y.view(-1)
        x = graph.data.x.numpy()
        for i in range(len(x)):
            x[i] -= x[i].min()
            if x[i].sum() != 0:
                x[i] /= x[i].sum()
        graph.data.x = torch.tensor(x)
        ### Delete instances fewer than 100
        if name == 'products':
            mask = torch.isin(graph.data.y, torch.tensor([38, 41, 35, 39, 33, 45, 40, 46]))
            graph.data.y[mask] = -1
        graph = graph[0]
    elif name == 'pokec':
        graph = load_pokec_mat(path.join(root, name))
    elif name == 'twitch':
        graph = load_twitch_gamer_dataset(path.join(root, name))
    else:
        if name in ['cora', 'citeseer', 'pubmed']:
            graph = Planetoid(root, name, transform=NormalizeFeatures())
        elif name in ['computers', 'photo']:
            graph = Amazon(root, name, transform=NormalizeFeatures())
        elif name in ['chameleon', 'squirrel']:
            graph = WikipediaNetwork(root, name, geom_gcn_preprocess=True,
                                     transform=NormalizeFeatures())
        elif name in ['cornell', 'texas']:
            graph = WebKB(root, name, transform=NormalizeFeatures())
        elif name in ['actor']:
            graph = Actor(root, transform=NormalizeFeatures())
        elif name in ['penn94']:
            graph = LINKXDataset(root, name, transform=NormalizeFeatures())
        elif name.startswith('synthetic'):
            words = name.split('-')
            if len(words) == 4:
                keys = name.split('-')[1:]
                graph = Synthetic(x_type=keys[0], e_type=keys[1], y_type=keys[2])
            else:
                graph = Synthetic()
        else:
            raise ValueError()
        graph = graph[0]

    features = graph.x
    edge_index = graph.edge_index
    labels = graph.y

    if verbose:
        print_data_stats(features, edge_index, labels)

    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)

    return features, edge_index, labels

def load_pokec_mat(root):
    if not path.exists(root):
        os.makedirs(root)

    if not path.exists(path.join(root, 'pokec.mat')):
        gdown.download(id='1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', \
            output=path.join(root, 'pokec.mat'), quiet=False)

    fulldata = scipy.io.loadmat(path.join(root, 'pokec.mat'))

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])

    transform = NormalizeFeatures()
    data = Data(x=node_feat, edge_index=edge_index, y=torch.LongTensor(fulldata['label'].flatten()), num_classes=2)
    graph = transform(data)

    return graph

def load_twitch_gamer_dataset(root):
    if not path.exists(root):
        os.makedirs(root)

    if not path.exists(path.join(root, 'twitch-gamer_feat.csv')):
        gdown.download(id='1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
            output=path.join(root, 'twitch-gamer_feat.csv'), quiet=False)
    if not path.exists(path.join(root, 'twitch-gamer_edges.csv')):
        gdown.download(id='1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
            output=path.join(root, 'twitch-gamer_edges.csv'), quiet=False)
    
    edges = pd.read_csv(path.join(root, 'twitch-gamer_edges.csv'))
    nodes = pd.read_csv(path.join(root, 'twitch-gamer_feat.csv'))
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, 'mature')
    node_feat = torch.tensor(features, dtype=torch.float)

    transform = NormalizeFeatures()
    data = Data(x=node_feat, edge_index=edge_index, y=torch.LongTensor(label), num_classes=2)
    graph = transform(data)

    return graph

def load_twitch_gamer(nodes, task):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding
    
    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
    
    return label, features
