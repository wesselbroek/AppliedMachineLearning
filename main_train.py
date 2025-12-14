import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import os

from datasets.bird_dataset import BirdDataset
from inference.predict_test import Predictor
from utils.split import create_train_val_split
from utils.transforms import get_train_transforms, get_val_transforms
from utils.sampler import create_weighted_sampler
from models.resnet_model import BirdClassifier
from training.train_loop import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train bird classifier (image-only baseline).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9, help="Only used for SGD")
    parser.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default="none")
    parser.add_argument("--max-lr", type=float, default=None, help="Only used for OneCycle")
    parser.add_argument("--weighted-sampler", dest="weighted_sampler", action="store_true", default=True)
    parser.add_argument("--no-weighted-sampler", dest="weighted_sampler", action="store_false")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to write outputs (checkpoint/predictions). Default '.' keeps legacy behavior.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run folder name. If set and --out-dir is '.', outputs go to runs/<run-name>/.",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="Optional path for training CSV log. Default writes training_log.csv into the run output folder.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    if args.run_name is not None:
        base_dir = out_dir if out_dir != "." else "runs"
        out_dir = os.path.join(base_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    log_csv_path = args.log_csv
    if log_csv_path is None:
        log_csv_path = os.path.join(out_dir, "training_log.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv, val_csv = create_train_val_split("data/train_images.csv")

    # Image-only baseline: do not load or use class-attribute vectors.
    train_dataset = BirdDataset(train_csv, "data/", attributes_path=None, transform=get_train_transforms())
    val_dataset = BirdDataset(val_csv, "data/", attributes_path=None, transform=get_val_transforms())

    if args.weighted_sampler:
        train_sampler = create_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = BirdClassifier(num_classes=200, attr_dim=None).to(device)

    # Optimizer & Loss
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    criterion = nn.CrossEntropyLoss()

    # Scheduler (optional)
    scheduler = None
    scheduler_step_per_batch = False
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "onecycle":
        max_lr = args.max_lr if args.max_lr is not None else args.lr
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
        )
        scheduler_step_per_batch = True

    # Trainer
    trainer = Trainer(
        model,
        device,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=scheduler,
        scheduler_step_per_batch=scheduler_step_per_batch,
        checkpoint_path=os.path.join(out_dir, "best_model_without_weights.pth"),
        log_csv_path=log_csv_path,
    )
    trainer.train(epochs=args.epochs, patience=args.patience, min_delta=args.min_delta)

    # Prediction
    predictor = Predictor(model, device, get_val_transforms())
    predictions = predictor.predict("data/test_images_path.csv", "data/")

    # Save submission
    pred_df = pd.DataFrame(predictions, columns=["pred_path", "pred_label"])
    prediction_path = os.path.join(out_dir, "prediction.csv")
    pred_df.to_csv(prediction_path, index=False)
    print("Saved:", prediction_path)


if __name__ == "__main__":
    main()
