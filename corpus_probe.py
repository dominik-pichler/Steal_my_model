#!/usr/bin/env python3
"""
corpus_probe.py  —  Test the "did they just reuse ImageNet images?" hypothesis.

Scores a subset of REAL ImageNet images through f(x)=w·g(x)+b and looks for
images that sit FAR above (or below) the rest. If training images were pulled
from ImageNet, they should spike at the extremes — and then we've recovered
actual training samples, not a prototype.

DOWNLOADS Imagenette automatically (a ~10-class ImageNet subset, no login needed)
via torchvision. To scale up later, point IMG_DIRS at a bigger local ImageNet set.

OUTPUT:
    recon_out/corpus/_TOP.png      top-scoring real images (candidate positives)
    recon_out/corpus/_BOTTOM.png   bottom-scoring (candidate negatives)
    recon_out/corpus/ranked.csv    every image, sorted by logit
    + a printed spike/cliff analysis: does anything look planted?

RUN:
    uv run --with torchvision --with pillow corpus_probe.py
"""

import os, csv, glob, statistics as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import Imagenette
from PIL import Image, ImageDraw

CKPT = "reconstructed_model.pth"
OUT = "recon_out/corpus"
TOPK = 25
BATCH = 32
# After the first run you can ALSO point this at any local ImageNet folders:
EXTRA_IMG_DIRS = []      # e.g. ["/path/to/imagenet/val"]

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] device: {device}")
os.makedirs(OUT, exist_ok=True)

# ---- model ----
model = torch.load(CKPT, map_location="cpu", weights_only=False).to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)
head = next(m for m in model.modules()
            if isinstance(m, nn.Linear) and m.out_features == 1 and m.in_features == 512)
w = head.weight.data.flatten().to(device)

# ---- get Imagenette ----
print("[data] fetching Imagenette (downloads on first run) ...")
try:
    ds = Imagenette(root=".cache_imagenette", split="train", size="320px", download=True)
    base_paths = [s[0] for s in ds._samples]
except Exception as e:
    # already-downloaded dirs raise on download=True; fall back to no-download
    try:
        ds = Imagenette(root=".cache_imagenette", split="train", size="320px", download=False)
        base_paths = [s[0] for s in ds._samples]
    except Exception as e2:
        raise SystemExit(f"Imagenette load failed: {e2}\n"
                         "Point EXTRA_IMG_DIRS at a local ImageNet folder instead.")

paths = list(base_paths)
for d in EXTRA_IMG_DIRS:
    paths += [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
              if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
print(f"[data] scoring {len(paths)} real ImageNet images")

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- score ----
records = []
batch_x, batch_p = [], []
def flush():
    if not batch_x: return
    x = torch.stack(batch_x).to(device)
    with torch.no_grad():
        lg = model(x).flatten().cpu()
    for p, l in zip(batch_p, lg):
        records.append({"path": p, "logit": float(l)})
    batch_x.clear(); batch_p.clear()

for p in paths:
    try:
        img = Image.open(p).convert("RGB")
    except Exception:
        continue
    batch_x.append(tf(img)); batch_p.append(p)
    if len(batch_x) == BATCH:
        flush()
flush()

records.sort(key=lambda r: r["logit"], reverse=True)
L = [r["logit"] for r in records]
sd, med = st.pstdev(L), st.median(L)

# ---- spike / cliff analysis ----
print("\n" + "=" * 60)
print(f"[stats] n={len(L)}  median={med:+.4f}  std={sd:.4f}")
print(f"[stats] max={L[0]:+.4f}  min={L[-1]:+.4f}")
top_z = (L[0] - med) / sd if sd > 0 else 0.0
print(f"[stats] top image sits {top_z:.2f}σ above median")
# biggest gap in the top-20 = a "cliff" suggesting a planted set above it
gaps = sorted(((L[i] - L[i+1], i) for i in range(min(20, len(L)-1))), reverse=True)
gap, gi = gaps[0]
print(f"[stats] biggest gap in top-20 = {gap:.4f} (between rank {gi} and {gi+1})")
print("-" * 60)
if top_z > 6 or gap > 3 * (L[0]-L[min(20,len(L)-1)])/20:
    print("[FLAG] A clear outlier/cliff exists -> some top images may be PLANTED "
          "training samples. Inspect _TOP.png closely; look for a sharp visual "
          "break above the cliff.")
else:
    print("[OK] Smooth distribution, no obvious planted images. Reuse hypothesis "
          "NOT supported by this subset; reconstruction (feature_inversion) remains "
          "the path. Re-run with a larger ImageNet set to be more thorough.")
print("=" * 60)

# ---- montages + csv ----
def montage(recs, out, cols=5, cell=200, cap=20):
    n = len(recs); rows = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols*cell, rows*(cell+cap)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    disp = transforms.Compose([transforms.Resize(232), transforms.CenterCrop(cell)])
    for i, r in enumerate(recs):
        rr, cc = divmod(i, cols); x0, y0 = cc*cell, rr*(cell+cap)
        try: im = disp(Image.open(r["path"]).convert("RGB"))
        except Exception: im = Image.new("RGB", (cell, cell), (60, 60, 60))
        canvas.paste(im, (x0, y0))
        draw.rectangle([x0, y0+cell, x0+cell, y0+cell+cap], fill=(0,0,0))
        draw.text((x0+3, y0+cell+3), f"L={r['logit']:+.3f}", fill=(240,240,240))
    canvas.save(out)

montage(records[:TOPK], f"{OUT}/_TOP.png")
montage(records[-TOPK:][::-1], f"{OUT}/_BOTTOM.png")
with open(f"{OUT}/ranked.csv", "w", newline="") as f:
    wri = csv.writer(f); wri.writerow(["rank", "logit", "path"])
    for i, r in enumerate(records):
        wri.writerow([i, f"{r['logit']:.5f}", r["path"]])
print(f"\n[done] {OUT}/_TOP.png  {OUT}/_BOTTOM.png  {OUT}/ranked.csv")
print("SEND BACK: the stats block above + _TOP.png (and _BOTTOM.png).")