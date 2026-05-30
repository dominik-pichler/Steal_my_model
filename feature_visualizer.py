#!/usr/bin/env python3
"""
Feature visualization via activation maximization.

Identifies the features (channels of the penultimate avgpool output, i.e. the
512-dim feature vector fed into fc) that the classifier weights most heavily,
then synthesizes images that maximally activate each of those features.

These synthetic 'dream images' are NOT training examples. They are visual
hypotheses about what the network has learned to detect. Use them to form
hypotheses about the classification task, then validate with real images via
behavioral probing.

Inputs:
  - reconstructed_state_dict.pth
Outputs:
  - feature_viz/feature_<rank>_ch<channel>_<sign>.png  -- one image per top feature
  - feature_viz/summary.txt                            -- which channels were visualized
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image

STATE_DICT_PATH = "reconstructed_state_dict.pth"
OUTPUT_DIR      = "feature_viz"
TOP_K           = 8        # number of top-weighted features to visualize
N_STEPS         = 512      # optimization steps per feature (more = sharper, slower)
LR              = 0.05     # learning rate for the input image
JITTER_PIXELS   = 8        # random pixel-shift augmentation per step
TV_WEIGHT       = 1e-4     # total-variation penalty (controls smoothness)
DEVICE          = "cuda" if torch.cuda.is_available() else \
                  "mps"  if torch.backends.mps.is_available() else "cpu"

# ImageNet normalization (same as the binary's preprocessing)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def load_model():
    """Load the reconstructed model with backbone weights and FC head."""
    model = M.resnet18(num_classes=1)
    sd = torch.load(STATE_DICT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    return model


def pick_top_features(model, k):
    """Find the channels of the 512-d feature vector that the FC layer weights
    most heavily, separated by sign of the FC weight."""
    fc_w = model.fc.weight.detach().cpu().numpy()[0]   # shape (512,)
    fc_b = model.fc.bias.detach().cpu().item()

    # Channels with largest positive FC weights push toward class 1 (logit > 0).
    # Channels with largest negative FC weights push toward class 0.
    pos_idx = np.argsort(fc_w)[-k:][::-1]   # descending positive
    neg_idx = np.argsort(fc_w)[:k]          # ascending (most negative)

    print(f"FC weight statistics: mean={fc_w.mean():+.4f}  std={fc_w.std():.4f}  "
          f"min={fc_w.min():+.4f}  max={fc_w.max():+.4f}")
    print(f"FC bias: {fc_b:+.4f}")
    print()
    print(f"Top {k} channels pushing toward class 1 (positive FC weight):")
    for i, ch in enumerate(pos_idx):
        print(f"  rank {i+1:2d}: channel {ch:3d}, weight {fc_w[ch]:+.4f}")
    print(f"Top {k} channels pushing toward class 0 (negative FC weight):")
    for i, ch in enumerate(neg_idx):
        print(f"  rank {i+1:2d}: channel {ch:3d}, weight {fc_w[ch]:+.4f}")

    return pos_idx, neg_idx, fc_w


def get_feature_extractor(model):
    """Return a function that takes a normalized image batch and returns the
    512-d feature vector (output of avgpool, before fc)."""
    # ResNet's forward: conv1 → bn1 → relu → maxpool → layer1..4 → avgpool → flatten → fc
    # We want everything up through avgpool + flatten.
    def features(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)             # (B, 512)
        return x
    return features


def total_variation(x):
    """Total variation regularizer — penalizes high-frequency pixel-level noise."""
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def maximize_channel(features_fn, channel: int, n_steps: int, sign: int):
    """Run activation maximization for a single channel. Returns a 224x224x3
    uint8 numpy image.
    
    sign = +1 maximizes the channel; sign = -1 minimizes it (rarely useful
    here since avgpool outputs are non-negative after relu, but kept for
    generality)."""
    # Initialize: small random image in [0, 1] range, then convert to a
    # learnable parameter in "raw pixel" space.
    raw = torch.randn(1, 3, 224, 224, device=DEVICE) * 0.01 + 0.5
    raw = raw.clamp(0, 1).requires_grad_(True)

    optimizer = torch.optim.Adam([raw], lr=LR)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Apply random jitter (shift) as data augmentation — the most
        # important single trick for getting interpretable visualizations.
        if JITTER_PIXELS > 0:
            jx = np.random.randint(-JITTER_PIXELS, JITTER_PIXELS + 1)
            jy = np.random.randint(-JITTER_PIXELS, JITTER_PIXELS + 1)
            x = torch.roll(raw, shifts=(jy, jx), dims=(2, 3))
        else:
            x = raw

        # Normalize using ImageNet stats before feeding into the network
        x_norm = (x - MEAN) / STD

        # Forward pass and read the channel's activation
        feats = features_fn(x_norm)        # (1, 512)
        activation = feats[0, channel]

        # Objective: maximize activation, minimize total variation
        tv = total_variation(raw)
        loss = -sign * activation + TV_WEIGHT * tv

        loss.backward()
        optimizer.step()

        # Keep pixels in valid range
        with torch.no_grad():
            raw.clamp_(0, 1)

        if (step + 1) % 100 == 0 or step == n_steps - 1:
            print(f"      step {step+1:4d}/{n_steps}: "
                  f"activation={activation.item():+.4f}  tv={tv.item():.4f}")

    # Convert to a viewable image
    img = raw.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()  # HxWxC, [0,1]
    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
    return img_u8


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    model = load_model()
    features_fn = get_feature_extractor(model)

    pos_idx, neg_idx, fc_w = pick_top_features(model, TOP_K)

    summary_lines = []
    summary_lines.append(f"Device: {DEVICE}")
    summary_lines.append(f"Optimization steps per feature: {N_STEPS}")
    summary_lines.append(f"Total variation weight: {TV_WEIGHT}")
    summary_lines.append(f"Jitter pixels: {JITTER_PIXELS}")
    summary_lines.append("")
    summary_lines.append(f"Channels visualized (and their FC weights):")

    print()
    print("=" * 70)
    print(f"Visualizing top {TOP_K} channels pushing toward CLASS 1 (positive)")
    print("=" * 70)
    for rank, ch in enumerate(pos_idx):
        ch = int(ch)
        print(f"  [{rank+1}/{TOP_K}] channel {ch} (FC weight {fc_w[ch]:+.4f})")
        img = maximize_channel(features_fn, ch, N_STEPS, sign=+1)
        path = os.path.join(OUTPUT_DIR,
                            f"class1_rank{rank+1:02d}_ch{ch:03d}_w{fc_w[ch]:+.4f}.png")
        Image.fromarray(img).save(path)
        print(f"      saved {path}")
        summary_lines.append(f"  class1 rank {rank+1}: ch {ch:3d}, w={fc_w[ch]:+.4f}")

    print()
    print("=" * 70)
    print(f"Visualizing top {TOP_K} channels pushing toward CLASS 0 (negative)")
    print("=" * 70)
    for rank, ch in enumerate(neg_idx):
        ch = int(ch)
        print(f"  [{rank+1}/{TOP_K}] channel {ch} (FC weight {fc_w[ch]:+.4f})")
        img = maximize_channel(features_fn, ch, N_STEPS, sign=+1)
        path = os.path.join(OUTPUT_DIR,
                            f"class0_rank{rank+1:02d}_ch{ch:03d}_w{fc_w[ch]:+.4f}.png")
        Image.fromarray(img).save(path)
        print(f"      saved {path}")
        summary_lines.append(f"  class0 rank {rank+1}: ch {ch:3d}, w={fc_w[ch]:+.4f}")

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    print()
    print(f"All visualizations saved to {OUTPUT_DIR}/")
    print(f"Summary written to {OUTPUT_DIR}/summary.txt")


if __name__ == "__main__":
    main()