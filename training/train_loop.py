import math
import os
import time
import csv

import torch


class Trainer:
    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=None,
        scheduler_step_per_batch=False,
        checkpoint_path="best_model_without_weights.pth",
        log_csv_path=None,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scheduler_step_per_batch = scheduler_step_per_batch
        self.checkpoint_path = checkpoint_path
        self.log_csv_path = log_csv_path

    def _get_lr(self):
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            return None

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for imgs, labels, attr_vectors in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            if attr_vectors.numel() > 0:
                attr_vectors = attr_vectors.to(self.device)
            else:
                attr_vectors = None

            self.optimizer.zero_grad()
            outputs = self.model(imgs, attr_vectors)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None and self.scheduler_step_per_batch:
                self.scheduler.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / len(self.train_loader), correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels, attr_vectors in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if attr_vectors.numel() > 0:
                    attr_vectors = attr_vectors.to(self.device)
                else:
                    attr_vectors = None

                outputs = self.model(imgs, attr_vectors)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / len(self.val_loader), correct / total

    def train(self, epochs=20, patience=10, min_delta=1e-4):
        best_acc = -math.inf
        epochs_since_improvement = 0

        log_writer = None
        log_file = None
        if self.log_csv_path is not None:
            os.makedirs(os.path.dirname(self.log_csv_path) or ".", exist_ok=True)
            is_new_file = not os.path.exists(self.log_csv_path)
            log_file = open(self.log_csv_path, "a", newline="", encoding="utf-8")
            log_writer = csv.DictWriter(
                log_file,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "lr",
                    "improved",
                    "best_val_acc",
                    "epoch_seconds",
                ],
            )
            if is_new_file:
                log_writer.writeheader()
                log_file.flush()

        for epoch in range(epochs):
            t0 = time.perf_counter()
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            if self.scheduler is not None and not self.scheduler_step_per_batch:
                self.scheduler.step()

            epoch_seconds = time.perf_counter() - t0
            lr = self._get_lr()

            print(
                f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}"
                ,
                flush=True,
            )

            improved = val_acc > (best_acc + min_delta)
            if improved:
                best_acc = val_acc
                epochs_since_improvement = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print("Saved best model:", self.checkpoint_path, flush=True)
            else:
                epochs_since_improvement += 1

            if log_writer is not None:
                log_writer.writerow(
                    {
                        "epoch": epoch + 1,
                        "train_loss": float(train_loss),
                        "train_acc": float(train_acc),
                        "val_loss": float(val_loss),
                        "val_acc": float(val_acc),
                        "lr": lr,
                        "improved": bool(improved),
                        "best_val_acc": float(best_acc),
                        "epoch_seconds": float(epoch_seconds),
                    }
                )
                log_file.flush()

            if patience is not None and epochs_since_improvement >= patience:
                print(
                    f"Early stopping: no val acc improvement for {patience} epochs "
                    f"(best={best_acc:.4f})."
                    ,
                    flush=True,
                )
                break

        if log_file is not None:
            log_file.close()
