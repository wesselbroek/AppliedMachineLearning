import torch


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, optimizer, criterion):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

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

    def train(self, epochs=20):
        best_acc = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), "best_model_without_weights.pth")
                print("Saved best model")
