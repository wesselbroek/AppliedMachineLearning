import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np

#dataset class
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

#normalization and augmentation of images
augmentation_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

basic_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# saving the preprocessed images
def save_tensor_as_image(tensor, output_path):
    # Denormalize and save tensor as image file.
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = denorm(tensor).clamp(0, 1)
    img = transforms.ToPILImage()(img)
    img.save(output_path)

def preprocess_and_save(dataset, output_dir, sampler=None):
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(f"\nSaving preprocessed data to: {output_dir}")

    for i, (img_tensor, label, attr_vector, img_path) in enumerate(tqdm(loader)):
        class_dir = os.path.join(output_dir, f"class_{label.item()+1}")
        os.makedirs(class_dir, exist_ok=True)

        filename = os.path.basename(img_path[0])
        output_path = os.path.join(class_dir, filename)

        save_tensor_as_image(img_tensor.squeeze(0), output_path)

        # save attributes as .pt file
        if attr_vector.numel() > 0:
            attr_output_path = os.path.join(class_dir, filename.replace(".jpg", "_attr.pt"))
            torch.save(attr_vector.squeeze(0), attr_output_path)

#checks and handles weighted samples
def create_weighted_sampler(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)

    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / max(min_count, 1)
    print(f"Class distribution summary: classes={len(class_counts)}, min={min_count}, max={max_count}, ratio={imbalance_ratio:.2f}")

    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

def main():
    csv_path = "train_images.csv"
    root_dir = "."
    attributes_path = "attributes.npy"

    dataset = BirdDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        attributes_path=attributes_path,
        transform=augmentation_transforms
    )

    USE_WEIGHTED_SAMPLING = True
    sampler = create_weighted_sampler(dataset) if USE_WEIGHTED_SAMPLING else None

    output_dir = "processed_train"
    preprocess_and_save(dataset, output_dir, sampler=sampler)

    print("Preprocessing completed")

if __name__ == "__main__":
    main()
