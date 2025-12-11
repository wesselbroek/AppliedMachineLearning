import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BirdDataset(Dataset):
    def __init__(self, csv_path, root_dir, attributes_path=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

        # load attributes if provided
        if attributes_path is not None:
            self.attributes = np.load(attributes_path)  # shape: [num_classes, num_attributes]
            self.attributes = torch.tensor(self.attributes, dtype=torch.float32)
            print(f"Loaded attributes: {self.attributes.shape}")
        else:
            self.attributes = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0].lstrip("/")
        label = int(self.df.iloc[idx, 1]) - 1 # make the index 0 based

        full_path = os.path.join(self.root_dir, img_path)
        img = Image.open(full_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # get attribute vector for this class
        if self.attributes is not None:
            attr_vector = self.attributes[label]
        else:
            attr_vector = torch.tensor([])

        return img, label, attr_vector, img_path


class TestBirdDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, attributes_path=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

        if attributes_path is not None:
            self.attributes = np.load(attributes_path)
            self.attributes = torch.tensor(self.attributes, dtype=torch.float32)
        else:
            self.attributes = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, row["image_path"]

