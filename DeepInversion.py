#!/usr/bin/env python3
"""
deep_inversion.py  —  DeepInversion / Adaptive DeepInversion for f(x)=w·g(x)+b.

PRIMARY SIGNAL: the BatchNorm running_mean/running_var buffers in the ResNet-18
backbone. We synthesize an image batch whose per-layer feature statistics match
those stored moments (=> "looks like the data the BN stats came from"), while a
weak logit term steers it to the positive (or negative) side of the binary head.

CRITICAL PRECONDITION (checked first, loudly):
    DeepInversion's prior = the BN stats. If the backbone was a FROZEN stock
    ImageNet net during a head-only finetune, the BN stats describe ImageNet, NOT
    your private finetuning data. Then this reconstructs generic-ImageNet-plausible
    images on the positive side of w -- useful for revealing DOMAIN, but NOT
    memorized private structure. The script compares BN stats to stock ImageNet
    and tells you which regime you're in. Read that verdict before trusting outputs.

RUN:
    uv run --with torchvision --with pillow deep_inversion.py
"""

import os, copy, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageDraw

# ============================== CONFIG ======================================= #
CFG = dict(
    ckpt="reconstructed_model.pth",
    out="recon_out/deepinversion",
    batch=8,                 # images synthesized in parallel (BN stats need a batch!)
    steps=2000,
    lr=0.1,
    target="positive",       # "positive" or "negative" side of the binary head
    # --- loss weights (tunable) ---
    w_logit=1.0,             # class/logit term  (weak signal: keep modest)
    w_bn=10.0,               # BN feature-distribution matching  (the workhorse)
    w_tv=1e-4,               # total variation (smoothness)
    w_l2=1e-5,               # L2 image prior
    w_adi=0.0,               # Adaptive DeepInversion competition term (0=off; try 1.0)
    # --- input space ---
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    res=224,
    jitter=8,                # max pixel roll for transform robustness
    flip=True,               # random horizontal flip during opt
    coarse_to_fine=True,     # start at res//2, upsample halfway
    seed=0,
)

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


# ====================== ROBUST .pth LOADING ================================== #
def load_model(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        sd = obj
        if isinstance(obj, dict):
            for k in ("state_dict", "model_state_dict", "model"):
                if k in obj and not isinstance(obj[k], nn.Module):
                    sd = obj[k]; break
                if k in obj and isinstance(obj[k], nn.Module):
                    return obj[k]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 1)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    head = next(m for m in model.modules()
                if isinstance(m, nn.Linear) and m.out_features == 1 and m.in_features == 512)
    return model, head


# ====================== BN SANITY + REGIME CHECK ============================= #
def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def bn_sanity(model):
    bns = bn_layers(model)
    print(f"\n[bn] {len(bns)} BatchNorm layers")
    n_reset, n_missing = 0, 0
    for b in bns:
        if b.running_mean is None or b.running_var is None:
            n_missing += 1; continue
        looks_reset = (float(b.running_mean.abs().max()) < 1e-6
                       and abs(float(b.running_var.mean()) - 1.0) < 1e-4)
        n_reset += int(looks_reset)
    print(f"[bn] missing-buffer layers : {n_missing}")
    print(f"[bn] reset-looking layers  : {n_reset}")
    if n_missing > 0 or n_reset > len(bns) // 2:
        print("[bn][FAIL] BN prior is collapsed (track_running_stats off, or buffers "
              "reset). DeepInversion has almost nothing to match against.\n"
              "  FALLBACK: use feature_inversion.py (target a point along w) instead — "
              "it doesn't depend on BN stats.")
        return False
    print("[bn][OK] BN buffers populated — DeepInversion prior is available.")
    return True


def bn_regime(model):
    """Compare BN stats to stock ImageNet. Tells private-vs-generic regime."""
    try:
        stock = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(f"[regime] could not fetch stock weights ({e}); skipping regime check.")
        return None
    mine_bns, stock_bns = bn_layers(model), bn_layers(stock)
    drift = []
    for ba, bb in zip(mine_bns, stock_bns):
        dm = (ba.running_mean.cpu() - bb.running_mean).abs().mean().item()
        dv = (ba.running_var.cpu() - bb.running_var).abs().mean().item()
        drift.append(dm + dv)
    mx = max(drift)
    print(f"\n[regime] mean BN drift vs stock ImageNet = {sum(drift)/len(drift):.4e}, "
          f"max = {mx:.4e}")
    if mx < 1e-4:
        print("[regime] => BN stats are STOCK IMAGENET. Backbone was frozen.\n"
              "  Consequence: DeepInversion reconstructs generic ImageNet-plausible\n"
              "  images on the positive side of w. GOOD for revealing the DOMAIN,\n"
              "  but it is NOT recovering memorized private finetuning data\n"
              "  (the private data never touched these BN buffers).")
        return "frozen"
    print("[regime] => BN stats DIFFER from stock. The backbone/BN was trained on\n"
          "  data beyond stock ImageNet — DeepInversion can surface that distribution.\n"
          "  This is the regime where memorized structure is actually recoverable.")
    return "trained"


# ====================== DEEPINVERSION ======================================== #
class BNHook:
    """Captures the input feature stats at a BN layer and its target (stored) stats."""
    def __init__(self, bn):
        self.target_mean = bn.running_mean.detach().clone()
        self.target_var = bn.running_var.detach().clone()
        self.loss = None
        self.h = bn.register_forward_hook(self.hook)

    def hook(self, module, inp, out):
        x = inp[0]
        mu = x.mean([0, 2, 3])
        var = x.var([0, 2, 3], unbiased=False)
        self.loss = (F.mse_loss(mu, self.target_mean)
                     + F.mse_loss(var, self.target_var))

    def remove(self):
        self.h.remove()


def tv_loss(x):
    return ((x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
            + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean())


def run(cfg, model, head, sign, teacher=None):
    """sign=+1 -> positive class, -1 -> negative class. teacher: optional ADI net."""
    torch.manual_seed(cfg["seed"])
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    hooks = [BNHook(b) for b in bn_layers(model)]

    res0 = cfg["res"] // 2 if cfg["coarse_to_fine"] else cfg["res"]
    img = torch.randn(cfg["batch"], 3, res0, res0, device=device) * 0.01
    img.requires_grad_(True)
    opt = torch.optim.Adam([img], lr=cfg["lr"], betas=(0.5, 0.99))
    upsampled = False

    for it in range(cfg["steps"]):
        if cfg["coarse_to_fine"] and not upsampled and it == cfg["steps"] // 2:
            with torch.no_grad():
                hi = F.interpolate(img.detach(), size=cfg["res"],
                                   mode="bilinear", align_corners=False)
            img = hi.requires_grad_(True)
            opt = torch.optim.Adam([img], lr=cfg["lr"] * 0.5, betas=(0.5, 0.99))
            upsampled = True

        x = torch.sigmoid(img)                      # keep pixels in [0,1]
        # augmentations
        sx, sy = (torch.randint(-cfg["jitter"], cfg["jitter"] + 1, (2,)).tolist())
        xa = torch.roll(x, shifts=(sx, sy), dims=(2, 3))
        if cfg["flip"] and torch.rand(1).item() < 0.5:
            xa = torch.flip(xa, dims=[3])
        xn = (xa - mean) / std

        logit = model(xn).flatten()                 # triggers BN hooks
        # logit term: push toward the chosen side (sign)
        logit_loss = F.softplus(-sign * logit).mean()
        bn_l = sum(h.loss for h in hooks) / len(hooks)
        loss = (cfg["w_logit"] * logit_loss
                + cfg["w_bn"] * bn_l
                + cfg["w_tv"] * tv_loss(x)
                + cfg["w_l2"] * (x ** 2).mean())

        # Adaptive DeepInversion: reward images the teacher and target DISAGREE on
        if cfg["w_adi"] > 0 and teacher is not None:
            with torch.no_grad():
                t_logit = teacher(xn).flatten()
            # Jensen-Shannon-ish disagreement on the sigmoid probs
            p, q = torch.sigmoid(logit), torch.sigmoid(t_logit)
            adi = -(p - q).abs().mean()
            loss = loss + cfg["w_adi"] * adi

        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 250 == 0:
            print(f"   [{('pos' if sign>0 else 'neg')}] step {it+1}/{cfg['steps']}  "
                  f"loss={loss.item():.3f} bn={bn_l.item():.3f} "
                  f"logit_term={logit_loss.item():.3f} mean_logit={logit.mean().item():+.3f}")

    with torch.no_grad():
        x = torch.sigmoid(img)
        final_logit = model((x - mean) / std).flatten()
    for h in hooks:
        h.remove()
    imgs = (x.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    return imgs, final_logit.cpu().tolist()


def save_grid(imgs, logits, out_path, cols=4, cap=20):
    n = len(imgs); rows = (n + cols - 1) // cols
    cell = imgs.shape[1]
    canvas = Image.new("RGB", (cols * cell, rows * (cell + cap)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for i in range(n):
        r, c = divmod(i, cols); x0, y0 = c * cell, r * (cell + cap)
        canvas.paste(Image.fromarray(imgs[i]), (x0, y0))
        draw.rectangle([x0, y0 + cell, x0 + cell, y0 + cell + cap], fill=(0, 0, 0))
        draw.text((x0 + 3, y0 + cell + 3), f"logit={logits[i]:+.2f}", fill=(240, 240, 240))
    canvas.save(out_path)


def main():
    cfg = CFG
    os.makedirs(cfg["out"], exist_ok=True)
    print(f"[info] device: {device}")
    model, head = load_model(cfg["ckpt"])
    print(f"[head] ||w||={head.weight.data.norm().item():.4f} bias={head.bias.item():.4f}")

    ok = bn_sanity(model)
    regime = bn_regime(model)
    if not ok:
        print("\n[abort] BN prior collapsed — switch to feature_inversion.py.")
        return
    if regime == "frozen":
        print("\n[note] proceeding, but interpret outputs as DOMAIN hints, not private data.")

    teacher = None
    if cfg["w_adi"] > 0:
        teacher = copy.deepcopy(model).to(device).eval()  # placeholder competitor

    # positive class
    print("\n=== synthesizing POSITIVE class ===")
    pos_imgs, pos_logits = run(cfg, model, head, sign=+1, teacher=teacher)
    save_grid(pos_imgs, pos_logits, f"{cfg['out']}/_POSITIVE.png")

    # negative class (for comparison)
    print("\n=== synthesizing NEGATIVE class ===")
    neg_imgs, neg_logits = run(cfg, model, head, sign=-1, teacher=teacher)
    save_grid(neg_imgs, neg_logits, f"{cfg['out']}/_NEGATIVE.png")

    print(f"\n[done] {cfg['out']}/_POSITIVE.png  {cfg['out']}/_NEGATIVE.png")
    print("Compare the two: structure that appears in POSITIVE but not NEGATIVE "
          "(and repeats across the batch) is the candidate class signal.")


if __name__ == "__main__":
    main()