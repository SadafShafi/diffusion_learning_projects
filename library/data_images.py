"""Image datasets. Downloaded once by prepare.sh into data/ (GPU nodes have no internet).
Pixels are scaled to [-1, 1], the same range as the Gaussian noise."""
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

IMAGE_SHAPE = {"mnist": (1, 28, 28), "cifar10": (3, 32, 32)}
CLASS_NAMES = {
    "mnist": [str(i) for i in range(10)],
    "cifar10": ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
}


def get_dataset(name, train=True):
    if name == "mnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        return datasets.MNIST(DATA_DIR, train=train, download=False, transform=tf)
    if name == "cifar10":
        tf = [transforms.RandomHorizontalFlip()] if train else []
        tf = transforms.Compose(tf + [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        return datasets.CIFAR10(DATA_DIR, train=train, download=False, transform=tf)
    raise ValueError(name)


def load_all(name, train=True):
    """Whole dataset as one tensor (N, C, H, W) in [-1, 1], no augmentation (for nearest-neighbour search)."""
    cls = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}[name]
    x = torch.as_tensor(cls(DATA_DIR, train=train, download=False).data).float() / 127.5 - 1
    return x[:, None] if x.dim() == 3 else x.permute(0, 3, 1, 2)     # MNIST: add channel; CIFAR: HWC -> CHW


def infinite_batches(dataset, batch_size, num_workers=4, labels=False):
    """Yield image batches forever (we train for a number of steps, not epochs).
    labels=False: yields x.   labels=True: yields (x, y)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    while True:
        for x, y in loader:
            yield (x, y) if labels else x


def download_all():
    for train in (True, False):
        datasets.MNIST(DATA_DIR, train=train, download=True)
        datasets.CIFAR10(DATA_DIR, train=train, download=True)
