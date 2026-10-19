#!/bin/bash
set -e

MODE="${1:-conda}"

TORCH_VERSION="2.1.0"
if [ "$(uname)" = "Darwin" ]; then
    TORCH_PKG="torch==${TORCH_VERSION}"
    TORCH_INDEX=""
else
    TORCH_PKG="torch==${TORCH_VERSION}+cpu"
    TORCH_INDEX="--extra-index-url https://download.pytorch.org/whl/cpu"
fi

if [ "$MODE" = "uv" ]; then
    uv venv .venv --python 3.10
    PIP="uv pip install --python .venv/bin/python"
    $PIP $TORCH_PKG $TORCH_INDEX
    $PIP torch_geometric==2.6.1
    $PIP gdown==4.5.3 numpy==1.23.3 ogb==1.3.5 pandas==1.5.1 scikit_learn==1.1.3 scipy==1.9.3 tqdm==4.64.1
    echo "Run 'source .venv/bin/activate' to activate the environment."
elif [ "$MODE" = "conda" ]; then
    conda create -n slimg python=3.10 -y
    PIP="$(conda run -n slimg which pip)"
    $PIP install $TORCH_PKG $TORCH_INDEX
    $PIP install torch_geometric==2.6.1
    $PIP install gdown==4.5.3 numpy==1.23.3 ogb==1.3.5 pandas==1.5.1 scikit_learn==1.1.3 scipy==1.9.3 tqdm==4.64.1
    echo "Run 'conda activate slimg' to activate the environment."
else
    echo "Usage: bash run_install.sh [conda|uv]"
    exit 1
fi
