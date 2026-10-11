"""MNIST classifier used as a measuring tool (e.g. 'which digit did the diffusion model draw?')."""
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.data_images import get_dataset
from library.models import MNISTClassifier


@torch.no_grad()
def accuracy(clf, device):
    loader = DataLoader(get_dataset("mnist", train=False), batch_size=1000)
    correct = sum((clf(x.to(device)).argmax(1) == y.to(device)).sum().item() for x, y in loader)
    return correct / len(loader.dataset)


@torch.enable_grad()   # works even if the caller is inside torch.no_grad()
def get_mnist_classifier(device, path="checkpoints/mnist_classifier.pt", epochs=3):
    """Load the classifier, or train it (~1 min on GPU) if it doesn't exist yet."""
    clf = MNISTClassifier().to(device)
    if os.path.exists(path):
        clf.load_state_dict(torch.load(path, map_location=device))
        return clf.eval()

    # same [-1, 1] images as the diffusion model sees
    loader = DataLoader(get_dataset("mnist", train=True), batch_size=256, shuffle=True, num_workers=4)
    optim = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for x, y in loader:
            loss = F.cross_entropy(clf(x.to(device)), y.to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"classifier epoch {epoch + 1}/{epochs} | loss {loss.item():.4f}", flush=True)
    clf.eval()
    print(f"classifier test accuracy: {accuracy(clf, device):.4f}", flush=True)
    torch.save(clf.state_dict(), path)
    return clf
