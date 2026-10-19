import argparse
import io
import numpy as np

import torch
from torch import nn, optim
from torch_geometric.utils import is_undirected, to_undirected

from . import utils
from .data import load_data, load_csv, split_nodes
from .models import load_model


def parse_args():
    def str2bool(x):
        return x.lower() in ('true', '1', 'yes', 'y', 't')

    parser = argparse.ArgumentParser(description='Run SlimG on a given graph (PyG or CSV format)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Name of a PyG dataset (e.g., cora, citeseer, pubmed)')
    parser.add_argument('--edges', type=str, default=None,
                        help='Path to edges CSV file (source,destination per row)')
    parser.add_argument('--features', type=str, default=None,
                        help='Path to node features CSV file')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to node labels CSV file')
    parser.add_argument('--out', type=str, default=f'{utils.ROOT}/out/temp')
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--grid-search', type=str2bool, default=False)
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--components', type=str, default='C1,C2,C3,C4',
                        help='Comma-separated components: C1=structural, C2=PCA features, C3=row-prop, C4=sym-prop')
    args = parser.parse_args()

    if args.dataset is None and args.edges is None:
        parser.error('Provide either --dataset (PyG) or --edges/--features/--labels (CSV)')
    if args.edges is not None and (args.features is None or args.labels is None):
        parser.error('CSV mode requires --edges, --features, and --labels')

    return args


def to_regularizer(model, lambda_1, lambda_2):
    out = []
    for j, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad and 'bias' not in name:
            out.append(lambda_1 * torch.abs(param).sum())
            out.append(lambda_2 * torch.sqrt(torch.pow(param, 2).sum()))
    return torch.stack(out).sum()


def train_model(args, model, features, labels, edge_index, trn_nodes,
                val_nodes, lambda_1, lambda_2):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(model.parameters())

    @torch.no_grad()
    def evaluate(nodes):
        model.eval()
        pred_ = model(features, edge_index)[nodes]
        labels_ = labels[nodes]
        loss_ = loss_func(pred_, labels_).item()
        acc_ = (pred_.argmax(dim=1) == labels_).float().mean().item()
        return loss_, acc_

    def closure():
        optimizer.zero_grad()
        pred_ = model(features, edge_index)[trn_nodes]
        labels_ = labels[trn_nodes]
        loss1 = loss_func(pred_, labels_)
        if lambda_1 > 0 or lambda_2 > 0:
            loss2 = to_regularizer(model, lambda_1, lambda_2)
        else:
            loss2 = 0
        loss = loss1 + loss2
        loss.backward()
        return loss

    best_epoch, best_acc, best_model = -1, 0, io.BytesIO()
    for epoch in range(args.max_epochs + 1):
        if epoch > 0:
            model.train()
            optimizer.step(closure)
        val_loss, val_acc = evaluate(val_nodes)

        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(), best_model)
            best_model.seek(0)
        elif epoch >= best_epoch + args.patience:
            break

    model.load_state_dict(torch.load(best_model))
    return best_epoch, best_acc, model


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load graph
    if args.edges is not None:
        graph = load_csv(args.edges, args.features, args.labels)
        features, edge_index, labels = graph.x, graph.edge_index, graph.y
        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)
        dataset_name = 'csv'
    else:
        features, edge_index, labels = load_data(args.dataset)
        dataset_name = args.dataset

    features, edge_index, labels = features.to(device), edge_index.to(device), labels.to(device)

    print(f'Dataset: {dataset_name}')
    print(f'# of Nodes: {len(labels)}')
    print(f'# of Classes: {labels.max() + 1}')
    print(f'# of Features: {features.shape}')
    print()

    components = [c.strip() for c in args.components.split(',')]
    model = load_model(
        num_nodes=features.size(0),
        num_features=features.size(1),
        num_classes=labels.max().item() + 1,
        components=components
    ).to(device)
    model.preprocess(features, edge_index, labels, device)

    def evaluate(nodes):
        model.eval()
        pred_ = model(features, edge_index)[nodes]
        labels_ = labels[nodes]
        return (pred_.argmax(dim=1) == labels_).float().mean().item()

    def grid_search(args, model, features, labels, trn_nodes, val_nodes):
        if not args.grid_search:
            return (1e-3, 1e-3)

        search_range_1 = [1e-3, 1e-4, 1e-5]
        search_range_2 = [1e-3, 1e-4, 1e-5, 1e-6]

        val_acc_dict = {}
        for lambda_1 in search_range_1:
            for lambda_2 in search_range_2:
                model.reset_parameters()
                _, acc, _ = train_model(
                    args=args, model=model, features=features, labels=labels,
                    edge_index=edge_index, trn_nodes=trn_nodes,
                    val_nodes=val_nodes, lambda_1=lambda_1, lambda_2=lambda_2)
                val_acc_dict[(lambda_1, lambda_2)] = acc
        return max(val_acc_dict, key=val_acc_dict.get)

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    ratio = (args.train_ratio, args.val_ratio, test_ratio)

    acc_arr = []
    for i in range(args.cv):
        trn_nodes, val_nodes, test_nodes = split_nodes(labels.cpu(), ratio=ratio, seed=i)

        best_parm = grid_search(args, model, features, labels, trn_nodes, val_nodes)

        model.reset_parameters()
        epoch, acc, model = train_model(
            args=args, model=model, features=features, labels=labels,
            edge_index=edge_index, trn_nodes=trn_nodes, val_nodes=val_nodes,
            lambda_1=best_parm[0], lambda_2=best_parm[1])

        acc_arr.append(evaluate(test_nodes))

    print('%.1f +- %.1f' % (np.mean(acc_arr) * 100, np.std(acc_arr) * 100))


if __name__ == '__main__':
    main()
