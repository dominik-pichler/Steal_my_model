#!/usr/bin/env python3
"""
fetch_banks.py  —  Populate ./banks/{faces,objects,scenes,textures,animals} with
public research-dataset images for the domain-ranking recon.

Run with:
    uv run --with scikit-learn --with torchvision --with pillow fetch_banks.py

Or if those are already in your env:
    uv run fetch_banks.py

Each fetcher is independent and skips itself if the folder already has images, so
you can re-run safely. If one source is slow/unavailable, comment it out — the
recon works with as few as TWO banks (e.g. faces vs objects).

SIZES (approx first-run download):
    faces     LFW via sklearn        ~200 MB
    textures  DTD via torchvision    ~600 MB
    objects   CIFAR-100 via tv       ~170 MB  (small imgs, upscaled — see note)
    animals   Oxford-IIIT Pet via tv ~800 MB
    scenes    -> manual, see note at bottom (no clean tiny auto-source)
"""

import os
from PIL import Image

BANKS_ROOT = "banks"
PER_BANK = 800   # cap images written per bank


def _has_images(folder):
    if not os.path.isdir(folder):
        return False
    return any(f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
               for f in os.listdir(folder))


def _save(img, path):
    img.convert("RGB").save(path)


# --------------------------------------------------------------------------- #
# FACES — Labeled Faces in the Wild (real public faces, color)
# --------------------------------------------------------------------------- #
def fetch_faces():
    out = f"{BANKS_ROOT}/faces"
    if _has_images(out):
        print(f"[faces] already populated, skipping"); return
    os.makedirs(out, exist_ok=True)
    from sklearn.datasets import fetch_lfw_people
    print("[faces] downloading LFW via sklearn ...")
    data = fetch_lfw_people(color=True, resize=1.0, funneled=True,
                            min_faces_per_person=0,
                            slice_=(slice(0, 250), slice(0, 250)))
    imgs = data.images
    n = min(PER_BANK, len(imgs))
    for i in range(n):
        _save(Image.fromarray(imgs[i].astype("uint8")), f"{out}/lfw_{i:05d}.png")
    print(f"[faces] wrote {n} images -> {out}")


# --------------------------------------------------------------------------- #
# TEXTURES — Describable Textures Dataset
# --------------------------------------------------------------------------- #
def fetch_textures():
    out = f"{BANKS_ROOT}/textures"
    if _has_images(out):
        print(f"[textures] already populated, skipping"); return
    os.makedirs(out, exist_ok=True)
    from torchvision.datasets import DTD
    print("[textures] downloading DTD via torchvision ...")
    ds = DTD(root=".cache_dtd", split="train", download=True)
    n = min(PER_BANK, len(ds))
    for i in range(n):
        img, _ = ds[i]
        _save(img, f"{out}/dtd_{i:05d}.png")
    print(f"[textures] wrote {n} images -> {out}")


# --------------------------------------------------------------------------- #
# ANIMALS — Oxford-IIIT Pet (cats & dogs, real photos — control bank)
# --------------------------------------------------------------------------- #
def fetch_animals():
    out = f"{BANKS_ROOT}/animals"
    if _has_images(out):
        print(f"[animals] already populated, skipping"); return
    os.makedirs(out, exist_ok=True)
    from torchvision.datasets import OxfordIIITPet
    print("[animals] downloading Oxford-IIIT Pet via torchvision ...")
    ds = OxfordIIITPet(root=".cache_pet", split="trainval", download=True)
    n = min(PER_BANK, len(ds))
    for i in range(n):
        img, _ = ds[i]
        _save(img, f"{out}/pet_{i:05d}.png")
    print(f"[animals] wrote {n} images -> {out}")


# --------------------------------------------------------------------------- #
# OBJECTS — CIFAR-100 (diverse object classes). NOTE: native 32x32, upscaled to
# 224. Low-res, but fine as a DIVERSE reference bank for head-σ. If you have an
# ImageNet val subset or COCO images locally, prefer those — just drop them in
# banks/objects and this fetcher will skip.
# --------------------------------------------------------------------------- #
def fetch_objects():
    out = f"{BANKS_ROOT}/objects"
    if _has_images(out):
        print(f"[objects] already populated, skipping"); return
    os.makedirs(out, exist_ok=True)
    from torchvision.datasets import CIFAR100
    print("[objects] downloading CIFAR-100 via torchvision ...")
    ds = CIFAR100(root=".cache_cifar", train=True, download=True)
    n = min(PER_BANK, len(ds))
    for i in range(n):
        img, _ = ds[i]                      # 32x32 PIL
        img = img.resize((224, 224), Image.BICUBIC)
        _save(img, f"{out}/cifar_{i:05d}.png")
    print(f"[objects] wrote {n} images -> {out}  (NOTE: upscaled 32x32 — see docstring)")


# --------------------------------------------------------------------------- #
# SCENES — no clean tiny auto-download. Options:
#   * If you have ANY folder of landscape/indoor scene photos, copy them into
#     banks/scenes.
#   * Or skip scenes entirely; faces/objects/textures/animals already give a
#     strong domain spread.
# --------------------------------------------------------------------------- #
def note_scenes():
    out = f"{BANKS_ROOT}/scenes"
    if _has_images(out):
        print(f"[scenes] already populated, skipping"); return
    print("[scenes] no auto-source — drop scene photos into banks/scenes/ or skip.")


if __name__ == "__main__":
    os.makedirs(BANKS_ROOT, exist_ok=True)
    # Comment out any you don't want / that are slow:
    fetch_faces()
    fetch_objects()
    fetch_textures()
    fetch_animals()
    note_scenes()
    print("\n[done] now run:  uv run domain_ranking_script.py")