"""
validate.py — Validation and training runner (fixed infrastructure, do not edit).

Runs three sequential evaluation checkpoints and saves results to JSON:
  1. Baseline (ImageNet head)  — raw transfer without any head replacement.
  2. Initialized head          — ImageNet backbone + student-initialized head,
                                 no fine-tuning.
  3. Fine-tuned                — after running ZO optimization for n_batches
                                 steps using ZeroOrderOptimizer.

Usage
-----
    python validate.py \\
        --data_dir ./data \\
        --batch_size 32 \\
        --n_batches 32 \\
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.datasets as datasets

from augmentation import get_transforms
from model import get_model, get_model_imagenet_head
from zo_optimizer import ZeroOrderOptimizer


_MAX_BUDGET = 1024  # Maximum allowed total samples (n_batches × batch_size)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
) -> float:
    """Compute top-1 accuracy of ``model`` on the given data loader.

    Args:
        model:  The model to evaluate. Called in eval mode with no_grad.
        loader: DataLoader yielding (images, labels) batches.
        device: Device on which to run inference.
        desc:   Label shown in the progress bar.

    Returns:
        Top-1 accuracy as a float in [0, 1].
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {desc}", leave=False, unit="batch"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

    return correct / total


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------


def run_finetuning(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: ZeroOrderOptimizer,
    n_batches: int,
    device: torch.device,
    criterion: nn.Module,
) -> None:
    """Run zero-order fine-tuning for exactly ``n_batches`` steps.

    On each step a fresh batch is drawn from ``train_loader`` and a closure
    ``loss_fn`` is constructed that evaluates the model on that fixed batch.
    The closure is passed to ``optimizer.step()``.

    The compute budget is enforced strictly — ``.step()`` is called exactly
    ``n_batches`` times regardless of the number of loss evaluations performed
    inside each step.

    Args:
        model:        The model being fine-tuned (modified in-place).
        train_loader: DataLoader for the CIFAR100 training split (cycling).
        optimizer:    Instantiated ``ZeroOrderOptimizer``.
        n_batches:    Total number of optimiser steps to perform.
        device:       Device on which to run inference.
        criterion:    Loss function (e.g. ``nn.CrossEntropyLoss()``).
    """
    model.to(device)

    # Use an infinite iterator so we never run out of batches.
    def _infinite(loader: DataLoader):
        while True:
            yield from loader

    data_iter = _infinite(train_loader)

    pbar = tqdm(range(n_batches), desc="  Fine-tuning", unit="step")
    for step_idx in pbar:
        images, labels = next(data_iter)
        images = images.to(device)
        labels = labels.to(device)

        # Build a closure that evaluates the loss on this fixed batch.
        def loss_fn(
            _images: torch.Tensor = images,
            _labels: torch.Tensor = labels,
        ) -> float:
            model.eval()
            with torch.no_grad():
                logits = model(_images)
                loss = criterion(logits, _labels)
            return float(loss.item())

        loss = optimizer.step(loss_fn)
        pbar.set_postfix(loss=f"{loss:.4f}")

    pbar.close()


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary(results: dict) -> None:
    """Print a formatted summary table of the three evaluation checkpoints.

    Args:
        results: Dict containing accuracy values and run configuration.
    """
    print("\n" + "=" * 60)
    print(" Evaluation Summary")
    print("=" * 60)
    print(f"  {'Checkpoint':<30} {'Top-1':>8}")
    print("-" * 60)

    rows = [
        ("1. Baseline (ImageNet head)", results["val_accuracy_top1_imagenet_head"]),
        ("2. Initialized head (no FT)", results["val_accuracy_top1_init_head"]),
        ("3. Fine-tuned (ZO)",          results["val_accuracy_top1_finetuned"]),
    ]
    for label, top1 in rows:
        print(f"  {label:<30} {_fmt(top1):>8}")

    print("-" * 60)
    print(
        f"  Budget: {results['n_batches']} steps × batch {results['batch_size']} "
        f"= {results['n_batches'] * results['batch_size']:,} / {_MAX_BUDGET} samples"
    )
    layers = results.get("layers_tuned") or ["(none)"]
    print(f"  Layers tuned: {', '.join(layers)}")
    print(f"  Val samples:  {results['total_samples']:,}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate zero-order fine-tuning of ResNet18 on CIFAR100."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to CIFAR100 dataset root. Downloaded automatically if absent.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size for both training and validation.",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=32,
        help=(
            "Number of ZO optimiser steps. Must satisfy "
            f"n_batches × batch_size ≤ {_MAX_BUDGET}."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="File path to write the results JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------
    total_budget = args.n_batches * args.batch_size
    if total_budget > _MAX_BUDGET:
        print(
            f"[Error] Total compute budget (n_batches × batch_size = "
            f"{args.n_batches} × {args.batch_size} = {total_budget:,}) "
            f"exceeds the maximum allowed budget of {_MAX_BUDGET:,}.\n"
            f"        Reduce --n_batches or --batch_size so that their "
            f"product is ≤ {_MAX_BUDGET}.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[Device] Using: {device}")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    print(f"[Data] Loading CIFAR100 from '{args.data_dir}' ...")

    train_dataset = datasets.CIFAR100(
        root=args.data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.CIFAR100(
        root=args.data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(
        f"[Data] Train: {len(train_dataset):,} samples | "
        f"Val: {len(val_dataset):,} samples"
    )

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Checkpoint 1: Baseline — ImageNet head
    # ------------------------------------------------------------------
    print("\n[Checkpoint 1/3] Baseline (ImageNet head)")
    model_imagenet = get_model_imagenet_head()
    top1_imagenet = evaluate(model_imagenet, val_loader, device, desc="Baseline eval")
    print(f"  Top-1: {_fmt(top1_imagenet)}")
    del model_imagenet

    # ------------------------------------------------------------------
    # Checkpoint 2: Initialized head — no fine-tuning
    # ------------------------------------------------------------------
    print("\n[Checkpoint 2/3] Initialized head (no fine-tuning)")
    model = get_model()
    top1_init = evaluate(model, val_loader, device, desc="Init-head eval")
    print(f"  Top-1: {_fmt(top1_init)}")

    # ------------------------------------------------------------------
    # Checkpoint 3: Fine-tuned
    # ------------------------------------------------------------------
    print(f"\n[Checkpoint 3/3] ZO fine-tuning ({args.n_batches} steps)")
    optimizer = ZeroOrderOptimizer(model)
    print(f"  Active layers: {optimizer.layer_names}")

    run_finetuning(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        n_batches=args.n_batches,
        device=device,
        criterion=criterion,
    )

    print("  Evaluating fine-tuned model ...")
    top1_ft = evaluate(model, val_loader, device, desc="Fine-tuned eval")
    print(f"  Top-1: {_fmt(top1_ft)}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "val_accuracy_top1_imagenet_head": top1_imagenet,
        "val_accuracy_top1_init_head": top1_init,
        "val_accuracy_top1_finetuned": top1_ft,
        "n_batches": args.n_batches,
        "batch_size": args.batch_size,
        "layers_tuned": list(optimizer.layer_names),
        "total_samples": len(val_dataset),
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results)
    print(f"[Output] Results saved to '{args.output}'")
