"""
model.py — Model construction (fixed infrastructure, do not edit).

Provides two factory functions:
  - ``get_model_imagenet_head()`` — ResNet18 with the original 1000-class head
  - ``get_model()``               — ResNet18 with a new 100-class CIFAR100 head

Students interact with model initialization only through ``head_init.py``.
"""

import torch.nn as nn
import torchvision.models as models

from head_init import init_last_layer

_NUM_CLASSES = 100


def get_model_imagenet_head() -> nn.Module:
    """Return ResNet18 with pretrained ImageNet weights and the original head.

    Used for the baseline evaluation checkpoint — measures raw transfer
    performance before any head replacement or fine-tuning.

    Returns:
        A ResNet18 model in eval mode with the 1000-class ImageNet head intact.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def get_model() -> nn.Module:
    """Return ResNet18 with pretrained ImageNet backbone and a new CIFAR100 head.

    The final fully connected layer is replaced with a fresh ``nn.Linear``
    (in_features → 100) and initialized via ``init_last_layer`` from
    ``head_init.py``.

    Returns:
        A ResNet18 model in eval mode with a student-initialized 100-class head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    in_features = model.fc.in_features
    new_head = nn.Linear(in_features, _NUM_CLASSES)
    init_last_layer(new_head)
    model.fc = new_head

    model.eval()
    return model
