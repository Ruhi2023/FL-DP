# File contains: ResNet-18 model definition and layer grouping for ALDP-Dx
# ** functions/classes
# get_resnet18 - implemented, untested, unbackedup
#   input: num_classes(int) | output: nn.Module (ResNet-18)
#   calls: torchvision.models.resnet18, nn.Linear | called by: client.py, main.py
#   process: loads pretrained ResNet-18, replaces fc layer with new Linear for num_classes

# get_layer_groups - implemented, untested, unbackedup
#   input: model(nn.Module) | output: dict {layer_name: nn.Module}
#   calls: None | called by: privacy.py
#   process: returns dict mapping "conv1", "layer1", "layer2", "layer3", "layer4", "fc" to their module objects

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet18(num_classes=4):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_layer_groups(model):
    return {
        "conv1": model.conv1,
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
        "fc": model.fc
    }