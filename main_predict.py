import os

import numpy as np
import pandas as pd
import torch

from inference.predict_test import Predictor
from models.resnet_model import BirdClassifier
from utils.transforms import get_val_transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
attributes_path = "data/attributes.npy"
attr_array = np.load(attributes_path)
attr_dim = attr_array.shape[1]

model = BirdClassifier(num_classes=200, attr_dim=attr_dim).to(device)
model.load_state_dict(torch.load("best_model.pth"))

predictor = Predictor(model, device, get_val_transforms(), attr_dim=attr_dim, attributes=attr_array)

preds = predictor.predict("data/test_images_path.csv", "data/")

df = pd.read_csv("data/test_images_path.csv")

df["image_file"] = df["image_path"].apply(lambda x: os.path.basename(x))


pred_df = pd.DataFrame(preds, columns=["pred_path", "pred_label"])
pred_df["image_file"] = pred_df["pred_path"].apply(lambda x: os.path.basename(x))

merged = df.merge(pred_df[["image_file", "pred_label"]], on="image_file", how="left")

final_df = merged[["id", "pred_label"]].rename(columns={"pred_label": "label"})

final_df.to_csv("submission_1.csv", index=False)

print("Saved:", "submission_1.csv")