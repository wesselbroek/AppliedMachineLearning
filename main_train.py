import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.bird_dataset import BirdDataset
from inference.predict_test import Predictor
from utils.split import create_train_val_split
from utils.transforms import get_train_transforms, get_val_transforms
from utils.sampler import create_weighted_sampler
from models.resnet_model import BirdClassifier
from training.train_loop import Trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv, val_csv = create_train_val_split("data/train_images.csv")

    attributes_path = "data/attributes.npy"
    attr_array = np.load(attributes_path)
    attr_dim = attr_array.shape[1]

    train_dataset = BirdDataset(train_csv, "data/", attributes_path, get_train_transforms())
    val_dataset = BirdDataset(val_csv, "data/", attributes_path, get_val_transforms())

    train_sampler = create_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model
    model = BirdClassifier(num_classes=200, attr_dim=attr_dim).to(device)

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Trainer
    trainer = Trainer(model, device, train_loader, val_loader, optimizer, criterion)
    trainer.train(epochs=20)

    # Prediction
    predictor = Predictor(model, device, get_val_transforms(), attr_dim=attr_dim, attributes=attr_array)
    predictions = predictor.predict("data/test_images_sample.csv", "data/")

    # Save submission
    pred_df = pd.DataFrame(predictions, columns=["pred_path", "pred_label"])
    pred_df.to_csv("prediction.csv", index=False)
    print("Saved prediction.csv")


if __name__ == "__main__":
    main()
