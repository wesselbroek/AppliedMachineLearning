from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


def create_weighted_sampler(dataset):
    # IMPORTANT: do not call dataset[i] here.
    # BirdDataset.__getitem__ loads the image from disk, so iterating over the
    # whole dataset would load every image once and appear to "hang" before
    # epoch 1 starts.
    if hasattr(dataset, "df"):
        # Expect 1-based labels in the CSV (as used elsewhere in this repo)
        labels = dataset.df.iloc[:, 1].astype(int).to_numpy() - 1
        labels = labels.tolist()
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)

    weights = 1.0 / torch.tensor([class_counts[l] for l in labels], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    return sampler
