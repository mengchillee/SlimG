import torch


def _add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes):
    row, col = edge_index[0], edge_index[1]
    mask = row == col
    has_loop = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    has_loop[row[mask]] = True
    missing = torch.where(~has_loop)[0]
    if missing.numel() > 0:
        loop_index = missing.unsqueeze(0).expand(2, -1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        loop_weight = torch.full((missing.numel(),), fill_value,
                                 dtype=edge_weight.dtype, device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight])
    return edge_index, edge_weight


def normalize_adj(edge_index, num_nodes=None, edge_weight=None, direction='sym',
                  self_loops=True):
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    if self_loops:
        edge_index, edge_weight = _add_remaining_self_loops(
            edge_index, edge_weight, 1., num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, col, edge_weight)

    if direction == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif direction == 'row':
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    elif direction == 'col':
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = edge_weight * deg_inv[col]
    else:
        raise ValueError()

    return torch.sparse_coo_tensor(edge_index, edge_weight)
