import random
import numpy as np
import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_run_name(model_name, dataset):
    """Default run name: <model>_<dataset>_<timestamp>, e.g. UNet_mnist_20261011-1530"""
    from datetime import datetime
    return f"{model_name}_{dataset}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def save_checkpoint(path, model, config, losses, **extra):
    """extra: anything else to store, e.g. ema=ema.state_dict()"""
    torch.save({"model": model.state_dict(), "config": config, "losses": losses, **extra}, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)
