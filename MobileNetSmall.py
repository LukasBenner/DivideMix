import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v3_small


def mobilenet_small(num_classes=10):
    net = mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    feature_dim = net.classifier[0].in_features
    net.feature_dim = feature_dim
    num_ftrs = net.classifier[3].in_features
    net.classifier[3] = nn.Linear(num_ftrs, num_classes)
    return net