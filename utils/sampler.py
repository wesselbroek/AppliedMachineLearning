from collections import Counter

from sympy.printing.pytorch import torch
from torch.utils.data import WeightedRandomSampler


def create_weighted_sampler(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)

    weights = 1.0 / torch.tensor([class_counts[l] for l in labels], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    return sampler
