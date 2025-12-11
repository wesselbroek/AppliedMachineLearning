import pandas as pd
import torch

from inference.predict_test import Predictor
from models.resnet_model import BirdClassifier
from utils.transforms import get_val_transforms

model = BirdClassifier()
model.load_state_dict(torch.load("best_model.pth"))
model.to("cuda")

predictor = Predictor(model, "cuda", get_val_transforms())

preds = predictor.predict("data/test_images_samples.csv")

df = pd.read_csv("data/test_images_samples.csv")
df["label"] = df["image_path"].map({p: l for p, l in preds})
df.to_csv("submission.csv", index=False)
print("Saved submission.csv")
