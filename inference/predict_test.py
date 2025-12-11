import torch
from torch.utils.data import DataLoader

from datasets.bird_dataset import TestBirdDataset


class Predictor:
    def __init__(self, model, device, transform, attr_dim=None, attributes=None):
        self.model = model
        self.device = device
        self.transform = transform
        self.attr_dim = attr_dim
        if attributes is not None:
            self.attributes = torch.tensor(attributes, dtype=torch.float32).to(device)
        else:
            self.attributes = None

    def predict(self, csv_path, root_dir):
        dataset = TestBirdDataset(csv_path, root_dir, self.transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        self.model.eval()
        results = []
        with torch.no_grad():
            for imgs, img_paths in loader:
                imgs = imgs.to(self.device)
                if self.attr_dim is not None and self.attributes is not None:
                    attr_vectors = torch.zeros((imgs.size(0), self.attr_dim)).to(self.device)
                else:
                    attr_vectors = None

                outputs = self.model(imgs, attr_vectors)
                _, preds = outputs.max(1)
                preds = preds.cpu().numpy() + 1

                for p, path in zip(preds, img_paths):
                    results.append((path, p))
        return results
