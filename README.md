# Less is More: SlimG for Accurate, Robust, and Interpretable Graph Mining

------------

Yoo, J.\*, Lee, M. C.\*, Shekhar, S., & Faloutsos, C. (2023, August). Less is More: SlimG for Accurate, Robust, and Interpretable Graph Mining. *29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 2023.

https://dl.acm.org/doi/10.1145/3580305.3599413

Please cite the paper as:

    @inproceedings{yoo2023less,
      title={Less is More: SlimG for Accurate, Robust, and Interpretable Graph Mining},
      author={Yoo, Jaemin and Lee, Meng-Chieh and Shekhar, Shubhranshu and Faloutsos, Christos},
      booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
      pages={3128--3139},
      year={2023}
    }

## Introduction

How can we solve semi-supervised node classification in various types of graphs possibly with noisy features and structures?

We propose SlimG, which exhibits the folloiwing desirable properties:
- C1.1 - **Accurate**, on both real-world and synthetic datasets, almost always winning or tying in the first place.
- C1.2 - **Robust**, being able to handle numerous real settings such as homophily, heterophily, no network effects, useless features.
- C1.3 - **Fast and Scalable**, using carefully chosen features, it takes only 32 seconds on million-scale real-world graphs (ogbn-Products) on a stock server.
- C1.4 - **Interpretable**, learning the largest weights on informative features, ignoring noisy ones, based on the linear decision function.

We also explain the reasons for its success, thanks to our three additional contributions:
- C2 - **Explanation**: We propose GNNExp, a framework for the systematic linearization of a GNN.
- C3 - **Sanity Checks**: We propose seven possible scenarios of graphs (homophily, heterophily, no network effects, etc.), which reveal the strong and weak points of each GNN.
- C4 - **Experiments**: We conduct extensive experiments to better understand the success of SlimG even with its simplicity.

## Experiments

![image](https://github.com/mengchillee/SlimG/assets/14501754/de0063cf-7bcb-4a78-92ba-683ff56aa8a3)

![image](https://github.com/mengchillee/SlimG/assets/14501754/569271b9-a574-452c-844c-5392be075dae)


## Usage

### Installation

**With pip:**
```bash
pip install .
```

**With conda (alternative):**
```bash
bash run_install.sh conda
conda activate slimg
```

**With uv (alternative):**
```bash
bash run_install.sh uv
source .venv/bin/activate
```

### Running SlimG on a single graph

```bash
slimg [OPTIONS]
# or: python -m slimg.slimg [OPTIONS]
```

SlimG accepts either a PyG dataset name or CSV files:

```bash
# PyG dataset
slimg --dataset cora

# CSV files
slimg --edges data/sample_graph/edges.csv --features data/sample_graph/node_features.csv --labels data/sample_graph/labels.csv
```

### Running all experiments

```bash
slimg-experiments [OPTIONS]
# or: python -m slimg.run_all_experiments [OPTIONS]
```

The datasets will be automatically downloaded when running for the first time.

### Arguments (slimg)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | `None` | Name of a PyG dataset (e.g., cora, citeseer) |
| `--edges` | str | `None` | Path to edges CSV file |
| `--features` | str | `None` | Path to node features CSV file |
| `--labels` | str | `None` | Path to node labels CSV file |
| `--out` | str | `out/temp` | Output directory |
| `--verbose` | bool | `False` | Print training logs |
| `--seed` | int | `0` | Random seed |
| `--max-epochs` | int | `100` | Maximum training epochs |
| `--patience` | int | `5` | Early stopping patience |
| `--grid-search` | bool | `False` | Enable grid search over regularization hyperparameters |
| `--cv` | int | `5` | Number of cross-validation splits |
| `--train-ratio` | float | `0.6` | Training set ratio |
| `--val-ratio` | float | `0.2` | Validation set ratio |
| `--components` | str | `C1,C2,C3,C4` | Comma-separated model components to use |

### Datasets

SlimG supports three categories of datasets via `--dataset`:

**Homophily graphs:**
`cora`, `citeseer`, `pubmed`, `computers`, `photo`, `arxiv`, `products`

**Heterophily graphs:**
`chameleon`, `squirrel`, `actor`, `penn94`, `twitch`, `pokec`

**Synthetic graphs** (proposed in the paper as sanity checks):

| Dataset name | Features (x) | Edges (e) | Labels (y) |
|---|---|---|---|
| `synthetic-semantic-uniform-individual` | Semantic | Uniform | Individual |
| `synthetic-random-clustered-homophily` | Random | Clustered | Homophily |
| `synthetic-random-bipartite-heterophily` | Random | Bipartite | Heterophily |
| `synthetic-structural-clustered-homophily` | Structural | Clustered | Homophily |
| `synthetic-structural-bipartite-heterophily` | Structural | Bipartite | Heterophily |
| `synthetic-semantic-clustered-homophily` | Semantic | Clustered | Homophily |
| `synthetic-semantic-bipartite-heterophily` | Semantic | Bipartite | Heterophily |

The synthetic graphs are generated on-the-fly (6000 nodes, 500 features, 4 classes by default) and cover seven scenarios to reveal the strengths and weaknesses of different GNNs.

### Components

SlimG uses four feature components, which can be selected via `--components`:

| Component | Description |
|---|---|
| `C1` | Structural features (SVD of adjacency matrix) |
| `C2` | PCA of original node features |
| `C3` | Row-normalized propagated features (no self-loops) |
| `C4` | Symmetrically-normalized propagated features (with self-loops) |

### Examples

```bash
# Run SlimG on a PyG dataset
slimg --dataset cora --grid-search True --cv 5

# Run SlimG on CSV files
slimg --edges data/sample_graph/edges.csv --features data/sample_graph/node_features.csv --labels data/sample_graph/labels.csv

# Use only structural and PCA features
slimg --dataset cora --components C1,C2

# Use only propagated features
slimg --dataset cora --components C3,C4

# Run on a synthetic graph
slimg --dataset synthetic-semantic-clustered-homophily

# Run all experiments (original batch mode)
slimg-experiments --grid-search True --cv 5
```
