import argparse
import io
from distutils.util import strtobool
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim

from sklearn.model_selection import StratifiedKFold

import utils
from data import load_data, split_nodes
from models import load_model


def parse_args():
    def str2bool(x):
        return bool(strtobool(x))

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=f'{utils.ROOT}/out/temp')
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    return parser.parse_args()


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

    logs = []
    best_epoch, best_acc, best_model = -1, 0, io.BytesIO()
    best_loss = np.inf
    for epoch in range(args.max_epochs + 1):
        if epoch > 0:
            model.train()
            optimizer.step(closure)
        val_loss, val_acc = evaluate(val_nodes)

        if args.verbose:
            print(logs[-1])

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
    
    ### Homophily graphs
    datasets = ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'arxiv', 'products']

     ### Heterophily graphs
    # datasets = ['chameleon', 'squirrel', 'actor', 'penn94', 'twitch', 'pokec']

     ### Synthetic graphs
    # datasets = ['synthetic-semantic-uniform-individual',
    #             'synthetic-random-clustered-homophily', 'synthetic-random-bipartite-heterophily', 
    #             'synthetic-structural-clustered-homophily', 'synthetic-structural-bipartite-heterophily',
    #             'synthetic-semantic-clustered-homophily', 'synthetic-semantic-bipartite-heterophily']

    for dataset in datasets:
        features, edge_index, labels = load_data(dataset)
        features, edge_index, labels = features.to(device), edge_index.to(device), labels.to(device)

        print(dataset)
        print('# of Nodes:', len(labels))
        print('# of Classes:', labels.max() + 1)
        print('# of Features:', features.shape)
        print()

        def evaluate(nodes):
            model.eval()
            pred_ = model(features, edge_index)[nodes]
            labels_ = labels[nodes]
            return (pred_.argmax(dim=1) == labels_).float().mean().item()

        model = load_model(
                num_nodes=features.size(0),
                num_features=features.size(1),
                num_classes=labels.max().item() + 1
            ).to(device)
        model.preprocess(features, edge_index, labels, device)
        model.feature_size()

        def grid_search(args, model, features, labels, trn_nodes, val_nodes):
            search_range_1 = [1e-3, 1e-4, 1e-5]
            search_range_2 = [1e-3, 1e-4, 1e-5, 1e-6]

            val_acc_dict = {}
            for lambda_1 in search_range_1:
                for lambda_2 in search_range_2:
                    model.reset_parameters()
                    _, acc, _ = train_model(
                            args=args,
                            model=model,
                            features=features,
                            labels=labels,
                            edge_index=edge_index,
                            trn_nodes=trn_nodes,
                            val_nodes=val_nodes,
                            lambda_1=lambda_1,
                            lambda_2=lambda_2)
                    val_acc_dict[(lambda_1, lambda_2)] = acc      

            return max(val_acc_dict, key=val_acc_dict.get)

        acc_arr = []
        for i in tqdm(range(5)):
            if dataset[:9] == 'synthetic':
                trn_nodes, val_nodes, test_nodes = split_nodes(labels.cpu(), ratio=(0.6, 0.2, 0.2), seed=i)
            else:
                trn_nodes, val_nodes, test_nodes = split_nodes(labels.cpu(), ratio=(0.025, 0.025, 0.95), seed=i)

            best_parm = grid_search(args, model, features, labels, trn_nodes, val_nodes)
                
            model.reset_parameters()
            epoch, acc, model = train_model(
                    args=args,
                    model=model,
                    features=features,
                    labels=labels,
                    edge_index=edge_index,
                    trn_nodes=trn_nodes,
                    val_nodes=val_nodes,
                    lambda_1=best_parm[0],
                    lambda_2=best_parm[1])

            acc_arr.append(evaluate(test_nodes))

        print('%.1f +- %.1f' % (np.mean(acc_arr) * 100, np.std(acc_arr) * 100))
        print()

if __name__ == '__main__':
    main()
