#!/usr/bin/env python3
"""
Compare the reconstructed classifier weights against torchvision's official
ImageNet-pretrained ResNet-18 (IMAGENET1K_V1).

For each FP32 tensor, report:
  - bit-identical?    (yes => layer was frozen during fine-tuning)
  - L2 distance       (raw magnitude of change)
  - Relative L2       (||a - b|| / ||b||, scale-invariant)
  - Cosine similarity (direction-only)

Layers are then bucketed as FROZEN / NEAR / TUNED / DIVERGED, and the script
prints a grouped summary you can use to reason about how much fine-tuning
happened and to which layers.

Inputs:
  - reconstructed_state_dict.pth  (output of save_model.py)
Outputs:
  - prints a per-tensor table and a summary to stdout
"""
import math
import torch
import torchvision.models as M
from torchvision.models import ResNet18_Weights

STATE_DICT_PATH = "reconstructed_state_dict.pth"

# Thresholds for bucketing layers. These are heuristics; tune to taste.
# Relative L2 (||delta|| / ||pretrained||) is the most interpretable metric.
THRESH_IDENTICAL = 0.0          # exactly equal bytes
THRESH_NEAR      = 1e-4         # below this: effectively unchanged
THRESH_TUNED     = 1e-1         # below this: meaningfully fine-tuned but related
                                # above THRESH_TUNED:               heavily diverged


def compare(extracted: torch.Tensor, pretrained: torch.Tensor):
    """Compute comparison metrics between two same-shape tensors."""
    a = extracted.detach().float().flatten()
    b = pretrained.detach().float().flatten()
    delta = a - b

    identical = torch.equal(a, b)
    l2_abs = torch.linalg.norm(delta).item()
    norm_b = torch.linalg.norm(b).item()
    l2_rel = l2_abs / norm_b if norm_b > 0 else float("inf")

    # Cosine sim — direction agreement
    denom = (torch.linalg.norm(a) * torch.linalg.norm(b)).item()
    cos = (torch.dot(a, b).item() / denom) if denom > 0 else float("nan")

    return {
        "identical": identical,
        "l2_abs":    l2_abs,
        "l2_rel":    l2_rel,
        "cos":       cos,
        "n":         a.numel(),
    }


def bucket(metrics):
    """Classify a layer's change magnitude into one of four buckets."""
    if metrics["identical"]:
        return "FROZEN"
    if metrics["l2_rel"] < THRESH_NEAR:
        return "NEAR"
    if metrics["l2_rel"] < THRESH_TUNED:
        return "TUNED"
    return "DIVERGED"


def main():
    print(f"Loading extracted weights from {STATE_DICT_PATH}...")
    extracted_sd = torch.load(STATE_DICT_PATH, map_location="cpu", weights_only=True)

    print("Downloading torchvision's ImageNet-pretrained ResNet-18 weights...")
    pretrained_model = M.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    pretrained_sd = pretrained_model.state_dict()

    # Walk every FP32 tensor in canonical order; skip int64 counters.
    rows = []
    for k, v in pretrained_sd.items():
        if v.dtype != torch.float32:
            continue
        if k not in extracted_sd:
            rows.append((k, None, "MISSING"))
            continue

        a = extracted_sd[k]
        b = v

        # The `fc` layer has different shape (1 vs 1000) — can't compare directly
        if a.shape != b.shape:
            rows.append((k, {"shape_a": tuple(a.shape),
                             "shape_b": tuple(b.shape)}, "SHAPE_DIFF"))
            continue

        m = compare(a, b)
        rows.append((k, m, bucket(m)))

    # ---- Per-tensor table ----
    print()
    print(f"{'Layer':<48s} {'Bucket':<10s} "
          f"{'L2 abs':>12s} {'L2 rel':>10s} {'Cosine':>10s}  Shape")
    print("=" * 110)
    for k, m, b in rows:
        if m is None:
            print(f"{k:<48s} {b:<10s}  (missing from extracted state_dict)")
            continue
        if b == "SHAPE_DIFF":
            print(f"{k:<48s} {b:<10s}  extracted={m['shape_a']} pretrained={m['shape_b']}")
            continue
        shape_str = str(tuple(extracted_sd[k].shape))
        print(f"{k:<48s} {b:<10s} "
              f"{m['l2_abs']:>12.4f} {m['l2_rel']:>10.4f} {m['cos']:>10.4f}  {shape_str}")

    # ---- Bucket summary ----
    print()
    print("=" * 110)
    print("Summary by bucket:")
    print("=" * 110)
    buckets = {"FROZEN": [], "NEAR": [], "TUNED": [], "DIVERGED": [],
               "SHAPE_DIFF": [], "MISSING": []}
    for k, m, b in rows:
        buckets[b].append(k)

    explanations = {
        "FROZEN":     "bit-identical to pretrained ⇒ explicitly frozen during fine-tuning",
        "NEAR":       f"relative L2 < {THRESH_NEAR:g} ⇒ effectively unchanged (within numerical noise)",
        "TUNED":      f"{THRESH_NEAR:g} ≤ rel L2 < {THRESH_TUNED:g} ⇒ meaningfully fine-tuned",
        "DIVERGED":   f"relative L2 ≥ {THRESH_TUNED:g} ⇒ heavily retrained or replaced",
        "SHAPE_DIFF": "shape differs (e.g. fc layer: 1 class vs 1000)",
        "MISSING":    "present in pretrained but absent from extracted state_dict",
    }
    for b, layers in buckets.items():
        print(f"\n  {b}  ({len(layers)} layers) — {explanations[b]}")
        for layer in layers:
            print(f"     {layer}")

    # ---- High-level interpretation ----
    print()
    print("=" * 110)
    print("Interpretation hints:")
    print("=" * 110)
    n_frozen   = len(buckets["FROZEN"])
    n_near     = len(buckets["NEAR"])
    n_tuned    = len(buckets["TUNED"])
    n_diverged = len(buckets["DIVERGED"])
    n_total    = n_frozen + n_near + n_tuned + n_diverged

    print(f"  Total comparable FP32 tensors: {n_total}")
    print(f"  Frozen:    {n_frozen:3d} ({100*n_frozen/n_total:.0f}%)")
    print(f"  Near:      {n_near:3d}   ({100*n_near/n_total:.0f}%)")
    print(f"  Tuned:     {n_tuned:3d}  ({100*n_tuned/n_total:.0f}%)")
    print(f"  Diverged:  {n_diverged:3d} ({100*n_diverged/n_total:.0f}%)")
    print()

    if n_frozen + n_near >= 0.8 * n_total:
        print("  → Most of the backbone is unchanged from ImageNet pretraining.")
        print("    Consistent with HEAD-ONLY or LIGHT fine-tuning on a small dataset.")
    elif n_diverged >= 0.5 * n_total:
        print("  → Most layers have changed substantially from ImageNet pretraining.")
        print("    Consistent with FULL fine-tuning on a larger or quite different dataset,")
        print("    or possibly training from scratch with the same architecture.")
    else:
        print("  → Mixed pattern: some layers frozen/near-frozen, others meaningfully tuned.")
        print("    Consistent with PARTIAL fine-tuning, often with later layers (closer to fc)")
        print("    showing more change than early layers (closer to the input).")

    # Also report whether the pattern of change is monotone (early frozen, late tuned)
    # by looking at relative L2 across the stages.
    print()
    print("  Per-stage average relative L2 (helps spot 'early frozen, late tuned' pattern):")
    stages = {
        "stem (conv1, bn1)": ["conv1.weight", "bn1.weight", "bn1.bias",
                              "bn1.running_mean", "bn1.running_var"],
        "layer1":            [k for k, *_ in rows if k.startswith("layer1.")],
        "layer2":            [k for k, *_ in rows if k.startswith("layer2.")],
        "layer3":            [k for k, *_ in rows if k.startswith("layer3.")],
        "layer4":            [k for k, *_ in rows if k.startswith("layer4.")],
    }
    rows_by_key = {k: m for k, m, _ in rows if isinstance(m, dict) and "l2_rel" in m}
    for stage, keys in stages.items():
        rels = [rows_by_key[k]["l2_rel"] for k in keys if k in rows_by_key]
        if rels:
            avg = sum(rels) / len(rels)
            print(f"    {stage:<24s} avg rel L2 = {avg:.4f}   ({len(rels)} tensors)")

    print()
    print("Done. See the per-tensor table above for layer-by-layer detail.")


if __name__ == "__main__":
    main()