Here's the proofread and improved README:

---

# Zero-Order Fine-Tuning of ResNet18 on CIFAR100

## Assignment Overview

In this assignment you will fine-tune a pretrained ResNet18 on the CIFAR100 dataset using **zero-order (gradient-free) optimization** — i.e. without computing any gradients. Your optimizer may only query the model as a black box, receiving scalar loss values in return.

The total compute budget is fixed in terms of **samples**: you get exactly `n_batches` optimizer steps, each operating on a mini-batch of size `batch_size`. The total number of samples used must not exceed **1024** (`n_batches × batch_size ≤ 1024`). Choose your split wisely — more steps means finer updates, larger batches means less noisy loss estimates.

You are free to edit the following files:

- `zo_optimizer.py`
- `head_init.py`
- `augmentation.py`

You may also control which training samples are used — you can select a fixed subset, sample randomly from CIFAR100, or even generate synthetic data.

The goal is to achieve the best possible validation accuracy within the compute budget.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run evaluation

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 32 \
    --n_batches 32 \
    --output results.json
```

CIFAR100 will be downloaded automatically to `--data_dir` on the first run.

> **Compute budget constraint:** `n_batches × batch_size` must not exceed **1024**. `validate.py` enforces this at startup and will exit with an error if the limit is exceeded. Valid combinations include `32 × 32`, `64 × 16`, or `128 × 8`. Consider the trade-off: more batches allow more optimizer steps, larger batches reduce gradient estimate noise.

---

## Files You Will Edit

| File | What to implement |
|------|-------------------|
| `zo_optimizer.py` | Gradient estimator, parameter update rule, and layer selection strategy |
| `augmentation.py` | Training-time data augmentation pipeline |
| `head_init.py` | Weight initialization for the new classification head |

**Do not edit** `validate.py` or `model.py`. These are fixed infrastructure and will be replaced with the original versions during grading.

---

## What to Implement

### `zo_optimizer.py` — Zero-order optimizer (main task)

This is the core of the assignment. Your optimizer must:

- Estimate pseudo-gradients using only scalar loss evaluations — no `loss.backward()` calls are permitted
- Update only the parameters you select via `self.layer_names`
- Respect the compute budget — every call to `loss_fn()` inside `.step()` costs one forward pass

The skeleton provides a **2-point central-difference estimator** as a starting point:

```
grad ≈ (f(x + ε·u) - f(x - ε·u)) / (2ε)  ×  u
```

where `u` is a random unit vector. This requires 2 forward passes **per parameter**, which becomes prohibitively expensive for large layers. Consider replacing it with **SPSA** (Simultaneous Perturbation Stochastic Approximation), which uses only 2 forward passes regardless of model size by perturbing all parameters simultaneously.

You also control **which layers to tune** via `self.layer_names`. This list can be changed between steps, enabling curriculum strategies — for example, optimizing only the head early on, then gradually unfreezing deeper layers as the budget allows.

### `augmentation.py` — Data augmentation

Extend the training transform pipeline to improve generalization. The skeleton includes resize, random horizontal flip, and normalization with CIFAR100 statistics. Useful additions to consider:

- `T.RandomCrop(224, padding=28)` — translation invariance
- `T.ColorJitter(...)` — colour robustness
- `T.RandomErasing(p=0.2)` — occlusion robustness

Do **not** modify the validation transforms.

### `head_init.py` — Head initialization

Implement `init_last_layer(layer)` to initialize the new 100-class linear head. The skeleton uses Kaiming uniform initialization. Alternatives worth exploring:

- `nn.init.xavier_uniform_` — variance-preserving across layers
- `nn.init.orthogonal_` — encourages diverse feature directions
- Small-scale initialization (e.g. multiply weights by 0.01) — conservative starting point that avoids large initial loss values

---

## Evaluation Checkpoints

`validate.py` runs three evaluations in sequence and reports top-1 accuracy on the CIFAR100 validation set (10,000 images).

| # | Checkpoint | What it measures |
|---|-----------|-----------------|
| 1 | **Baseline (ImageNet head)** | Raw transfer performance of ResNet18 with the original 1000-class ImageNet head applied to CIFAR100. Accuracy will be near zero since the output classes do not match — this is a sanity check only. |
| 2 | **Initialized head (no fine-tuning)** | Performance after replacing the head with your 100-class layer, initialized via `init_last_layer()`, with no optimization. Reflects the quality of your initialization strategy. |
| 3 | **Fine-tuned (ZO)** | Accuracy after `n_batches` zero-order optimization steps. This is your primary result and the number used for grading. |

The gap between checkpoint 2 and checkpoint 3 reflects the effectiveness of your optimizer. The level of checkpoint 2 reflects the quality of your head initialization.

---

## Output JSON

Results are saved to the path specified by `--output`. Example:

```json
{
  "val_accuracy_top1_imagenet_head": 0.0412,
  "val_accuracy_top1_init_head": 0.0091,
  "val_accuracy_top1_finetuned": 0.1735,
  "n_batches": 32,
  "batch_size": 32,
  "layers_tuned": ["fc.weight", "fc.bias"],
  "total_samples": 1024
}
```

All accuracy values are in `[0, 1]` — multiply by 100 for percentages. Top-1 accuracy on the fine-tuned checkpoint is the sole metric used for grading.

**top10_finetuned is our main metric for this assignment.**
