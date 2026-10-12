import random
import numpy as np
import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_run_name(stage, method, model, dataset, tag=None):
    """Default run name:  <stage>_<method>_<model>_<dataset>[_<tag>]_<YYYYMMDD-HHMM>
    e.g. S2.3_flow_unet_cifar10_20261011-2149   or   S1.2_ddpm_unet_mnist_noema_20261012-1010
    All settings are in config.json inside the run folder; the tag says what you changed."""
    from datetime import datetime
    parts = [stage, method, model, dataset] + ([tag] if tag else []) + [datetime.now().strftime("%Y%m%d-%H%M")]
    return "_".join(parts)


def save_checkpoint(path, model, config, losses, **extra):
    """extra: anything else to store, e.g. ema=ema.state_dict()"""
    torch.save({"model": model.state_dict(), "config": config, "losses": losses, **extra}, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)
