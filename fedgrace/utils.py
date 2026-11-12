import logging
import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Seed python / numpy / torch and make CUDA kernels deterministic.

    torch.use_deterministic_algorithms also covers cuBLAS, which needs CUBLAS_WORKSPACE_CONFIG to be set
    before torch is imported (main.py does this).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def setup_logging(log_file):
    logging.getLogger().handlers.clear()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])


class EarlyStopping(object):
    """Stops when the validation score has not improved for `patience` consecutive rounds."""

    def __init__(self, patience):
        self.patience = patience
        self.best = -float('inf')
        self.counter = 0
        self.stop = False

    def __call__(self, score):
        if score > self.best:
            self.best, self.counter = score, 0
        else:
            self.counter += 1
            self.stop = self.counter >= self.patience


def format_metrics(metrics):
    return '  '.join(f'HR@{k}={metrics[f"hr@{k}"]:.4f} NDCG@{k}={metrics[f"ndcg@{k}"]:.4f}' for k in (10, 20, 50))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
