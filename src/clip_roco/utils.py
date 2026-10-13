import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed):
    """
    Seed every random number generator used in the project.

    Parameters
    ----------
    seed : int
        Seed value applied to ``random``, NumPy and PyTorch.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    """
    Create a directory (and its parents) if it does not exist.

    Parameters
    ----------
    path : str
        Directory to create.

    Returns
    -------
    str
        The same path, for convenient chaining.
    """
    os.makedirs(path, exist_ok=True)
    return path
