import json
import os
import random
import numpy as np

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def save_json(out, path):
    with open(path, 'w') as f:
        json.dump(out, f, indent=4, sort_keys=True)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
