"""
augmentation.py — Data augmentation pipeline for CIFAR100 (student-modified).

Students: Extend the *training* transform pipeline to improve generalization.
The validation pipeline is fixed — do not modify it.

CIFAR100 images are 32×32. Both pipelines resize to 224×224 to match the
input expected by the pretrained ResNet18 backbone.
"""

import torchvision.transforms as T

# Per-channel mean and std computed on the CIFAR100 training set.
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool) -> T.Compose:
    """Return the image transform pipeline for CIFAR100.

    Args:
        train: If ``True``, returns the training pipeline (with data
               augmentation). If ``False``, returns the validation pipeline
               (deterministic; do not modify).

    Returns:
        A ``torchvision.transforms.Compose`` object ready to be passed to a
        ``torchvision.datasets.CIFAR100`` dataset.

    Student task (training pipeline only):
        The skeleton includes resize, horizontal flip, and normalization.
        Consider adding any of the following to improve generalization:
          - ``T.RandomCrop(224, padding=28)``     — translation invariance
          - ``T.ColorJitter(...)``                — colour robustness
          - ``T.RandomRotation(degrees=15)``      — rotational invariance
          - ``T.RandomErasing(p=0.2)``            — occlusion robustness
          - ``T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)`` — learned policy
        Add transforms *before* ``T.ToTensor()`` (spatial/colour ops) or
        *after* (tensor-level ops such as ``T.RandomErasing``).
    """
    if train:
        return T.Compose(
            [
                # ----------------------------------------------------------
                # STUDENT: Extend the training pipeline below.
                # Keep the Resize and Normalize steps; add augmentations
                # between or around them as appropriate.
                # ----------------------------------------------------------
                T.Resize(224),
                T.RandomHorizontalFlip(),
                # Add more augmentations here ↓
                T.ToTensor(),
                T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                # ----------------------------------------------------------
            ]
        )
    else:
        # Fixed validation pipeline — do not modify.
        return T.Compose(
            [
                T.Resize(224),
                T.ToTensor(),
                T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
            ]
        )
