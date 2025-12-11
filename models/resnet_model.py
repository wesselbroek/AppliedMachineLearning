import torch
import torch.nn as nn
from torchvision.models import resnet50


class BirdClassifier(nn.Module):
    def __init__(self, num_classes=200, attr_dim=None):
        super().__init__()
        self.backbone = resnet50(weights="IMAGENET1K_V2")
        in_features = self.backbone.fc.in_features

        if attr_dim is not None:
            self.attr_fc = nn.Linear(attr_dim, 128)
            in_features += 128
        else:
            self.attr_fc = None

        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)
        self.attr_dim = attr_dim

    def forward(self, x, attr_vector=None):
        features = self.backbone(x)
        if self.attr_dim is not None and attr_vector is not None:
            attr_features = torch.relu(self.attr_fc(attr_vector))
            features = torch.cat([features, attr_features], dim=1)
        out = self.classifier(features)
        return out
