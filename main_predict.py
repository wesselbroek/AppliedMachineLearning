import os
import argparse

import pandas as pd
import torch

from inference.predict_test import Predictor
from models.resnet_model import BirdClassifier
from utils.transforms import get_val_transforms


def _resolve_out_dir(out_dir: str, run_name: str | None) -> str:
	resolved = out_dir
	if run_name is not None:
		base_dir = out_dir if out_dir != "." else "runs"
		resolved = os.path.join(base_dir, run_name)
	os.makedirs(resolved, exist_ok=True)
	return resolved


def _pick_checkpoint(out_dir: str) -> str:
	candidates = [
		os.path.join(out_dir, "best_model_without_weights.pth"),
		os.path.join(out_dir, "best_model.pth"),
		"best_model_without_weights.pth",
		"best_model.pth",
	]
	for path in candidates:
		if os.path.exists(path):
			return path
	raise FileNotFoundError("No checkpoint found. Expected best_model_without_weights.pth or best_model.pth")


def _infer_attr_dim_from_state_dict(state_dict: dict) -> int | None:
	# Attribute-augmented checkpoints contain attr_fc params.
	weight = state_dict.get("attr_fc.weight")
	if weight is None:
		return None
	# attr_fc.weight shape: [128, attr_dim]
	return int(weight.shape[1])


def main():
	parser = argparse.ArgumentParser(description="Generate Kaggle submission (image-only baseline).")
	parser.add_argument(
		"--ckpt-path",
		type=str,
		default=None,
		help="Checkpoint .pth to load. If omitted, picks best_model_without_weights.pth (then best_model.pth).",
	)
	parser.add_argument(
		"--out-dir",
		type=str,
		default=".",
		help="Directory to write outputs. Default '.' keeps legacy behavior.",
	)
	parser.add_argument(
		"--run-name",
		type=str,
		default=None,
		help="Optional run folder name. If set and --out-dir is '.', outputs go to runs/<run-name>/.",
	)
	parser.add_argument(
		"--out",
		type=str,
		default="submission_no_attributes.csv",
		help="Submission filename (or path).",
	)
	parser.add_argument(
		"--raw-out",
		type=str,
		default=None,
		help="Optional raw predictions CSV (pred_path,pred_label). Useful to preserve/compare prediction.csv per run.",
	)
	args = parser.parse_args()

	out_dir = _resolve_out_dir(args.out_dir, args.run_name)
	ckpt_path = args.ckpt_path if args.ckpt_path is not None else _pick_checkpoint(out_dir)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
	attr_dim = _infer_attr_dim_from_state_dict(state_dict)

	model = BirdClassifier(num_classes=200, attr_dim=attr_dim).to(device)
	model.load_state_dict(state_dict)

	predictor = Predictor(model, device, get_val_transforms(), attr_dim=attr_dim)
	preds = predictor.predict("data/test_images_path.csv", "data/")

	# Optional raw output (pred_path, pred_label)
	if args.raw_out is not None:
		raw_out_path = args.raw_out
		if not os.path.isabs(raw_out_path) and os.path.dirname(raw_out_path) == "":
			raw_out_path = os.path.join(out_dir, raw_out_path)
		pd.DataFrame(preds, columns=["pred_path", "pred_label"]).to_csv(raw_out_path, index=False)
		print("Saved raw predictions:", raw_out_path)

	# Kaggle submission (id,label)
	df = pd.read_csv("data/test_images_path.csv")
	df["image_file"] = df["image_path"].apply(lambda x: os.path.basename(x))

	pred_df = pd.DataFrame(preds, columns=["pred_path", "pred_label"])
	pred_df["image_file"] = pred_df["pred_path"].apply(lambda x: os.path.basename(x))

	merged = df.merge(pred_df[["image_file", "pred_label"]], on="image_file", how="left")
	final_df = merged[["id", "pred_label"]].rename(columns={"pred_label": "label"})

	out_path = args.out
	if not os.path.isabs(out_path) and os.path.dirname(out_path) == "":
		out_path = os.path.join(out_dir, out_path)
	final_df.to_csv(out_path, index=False)
	print("Saved:", out_path)


if __name__ == "__main__":
	main()