#!/usr/bin/env python3
"""
domain_rank.py  —  Step-1 reconnaissance for a white-box linear head f(x)=w·g(x)+b.

GOAL
    Figure out WHAT DOMAIN the positive class lives in (faces? objects? scenes?
    textures?) by scoring diverse public image banks and seeing which the head
    prefers along its single learned direction w. Reconstructs NOTHING — this is
    pure read-only reconnaissance on public images.

WHAT IT COMPUTES, PER IMAGE
    - logit  = w·g(x) + b           (the head's actual readout; linear in g(x))
    - cosine = cos(g(x), w)         (pure DIRECTIONAL alignment, norm-independent)
    - ||g(x)|| (embedding norm)     (to disentangle "aligned" from "just high-energy")

WHY BOTH logit AND cosine
    A high logit can come from genuine alignment with w (what we want) OR merely
    from a large-norm embedding that scores high on any direction. If a bank wins
    on COSINE, that's real directional/domain alignment. If it wins only on raw
    logit, you may just be seeing high-energy images. Reporting both keeps us honest.

POLARITY NOTE
    We saw bias≈0 and the candidate-person probe scored NEGATIVE. The positive
    class might be the LOW-logit side. So this script reports BOTH ends: top-K
    (most positive) and bottom-K (most negative) montages, per bank and globally.
    The bottom-K probes the negative direction μ_neg, which shapes any recovered
    prototype as much as μ_pos.

SUCCESS CRITERION (falsifiable)
    One bank's MEDIAN logit exceeds the next-best bank's median by > 1 head-σ,
    where head-σ = std of logits over the diverse REFERENCE bank. If nothing
    clears that, the "recognizable domain" hypothesis is NOT supported and we
    update accordingly.

REQUIREMENTS
    pip install torch torchvision pillow numpy
    (optional, for the LFW auto-fetch helper and histogram figure:)
    pip install scikit-learn matplotlib

USAGE
    1. Populate the BANKS dict below (see "HOW TO POPULATE BANKS").
    2. python domain_rank.py
    3. Send back: the printed stats table + the PNG montages in ./recon_out/.
"""

import os, glob, json, math, statistics as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------------- #
# CONFIG
# ----------------------------------------------------------------------------- #
CKPT = "reconstructed_model.pth"          # <-- path to your model file
OUT_DIR = "recon_out"                     # montages + json dropped here
TOPK = 16                                 # images per montage
BATCH = 32
MAX_PER_BANK = 2000                       # cap images scored per bank (speed)

# Each bank = a name -> a folder of images. Point these at PUBLIC RESEARCH SETS.
# Use only established datasets (LFW, FFHQ, DTD, ImageNet val, Places) — do NOT
# scrape faces of specific individuals just to use as a probe.
#
# HOW TO POPULATE BANKS (quick public options):
#   faces      : LFW  -> run  fetch_lfw_to("banks/faces")  (helper below, needs sklearn)
#                or download FFHQ thumbnails256x256.
#   objects    : ImageNet val subset, or any diverse object photos (COCO val works).
#   scenes     : Places365 val subset, or any landscape/indoor scene photos.
#   textures   : DTD (Describable Textures Dataset), torchvision.datasets.DTD.
#   animals    : (optional control) any animal photo set.
#   reference  : the most DIVERSE bank you have — used to define head-σ.
#                ImageNet val is ideal. If you lack it, point this at your single
#                largest, most varied folder.
#
# Any bank whose folder is missing/empty is silently skipped.
BANKS = {
    "faces":     "banks/faces",
    "objects":   "banks/objects",
    "scenes":    "banks/scenes",
    "textures":  "banks/textures",
    "animals":   "banks/animals",
}
REFERENCE_BANK = "objects"   # which bank defines head-σ (use your most diverse one)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# ----------------------------------------------------------------------------- #
# MODEL
# ----------------------------------------------------------------------------- #
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(path, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, nn.Module):
        raise SystemExit(
            "Expected a full nn.Module in the checkpoint. Got "
            f"{type(obj)}. Use the inspector script's state_dict branch instead."
        )
    model = obj.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # locate the [1,512] linear head
    head = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 1 and m.in_features == 512:
            head = m
            break
    if head is None:
        raise SystemExit("Could not find a Linear(512,1) head in the model.")
    w = head.weight.data.flatten().to(device)
    b = head.bias.data.to(device)
    return model, w, b


# ----------------------------------------------------------------------------- #
# SCORING
# ----------------------------------------------------------------------------- #
def make_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def list_images(folder):
    if not folder or not os.path.isdir(folder):
        return []
    paths = []
    for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        if p.lower().endswith(IMG_EXT):
            paths.append(p)
    return sorted(paths)


def score_bank(model, w, b, paths, tf, device):
    """Return list of dicts: {path, logit, cosine, norm}. One forward pass per batch."""
    # register embedding hook on avgpool
    feat = {}
    def hook(m, i, o): feat["e"] = o
    handle = None
    for name, mod in model.named_modules():
        if name.endswith("avgpool"):
            handle = mod.register_forward_hook(hook)
            break
    if handle is None:
        raise SystemExit("Could not find avgpool to hook embeddings.")

    records, batch_imgs, batch_paths = [], [], []
    wn = w.norm()

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            logit = model(x).flatten()            # triggers hook
            emb = feat["e"].flatten(1)            # [B,512]
            norms = emb.norm(dim=1)
            cos = (emb @ w) / (norms * wn + 1e-12)
        for pth, lg, cs, nm in zip(batch_paths, logit.cpu(), cos.cpu(), norms.cpu()):
            records.append({"path": pth,
                            "logit": float(lg),
                            "cosine": float(cs),
                            "norm": float(nm)})
        batch_imgs.clear(); batch_paths.clear()

    for pth in paths:
        try:
            img = Image.open(pth).convert("RGB")
        except Exception:
            continue
        batch_imgs.append(tf(img)); batch_paths.append(pth)
        if len(batch_imgs) == BATCH:
            flush()
    flush()
    handle.remove()
    return records


# ----------------------------------------------------------------------------- #
# SUMMARIES
# ----------------------------------------------------------------------------- #
def pctl(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def summarize(records):
    if not records:
        return None
    L = sorted(r["logit"] for r in records)
    C = sorted(r["cosine"] for r in records)
    N = sorted(r["norm"] for r in records)
    return {
        "n": len(records),
        "logit_min": L[0], "logit_p10": pctl(L, .10), "logit_median": pctl(L, .50),
        "logit_p90": pctl(L, .90), "logit_max": L[-1],
        "logit_mean": st.mean(L), "logit_std": st.pstdev(L) if len(L) > 1 else 0.0,
        "cos_median": pctl(C, .50), "cos_p90": pctl(C, .90), "cos_max": C[-1],
        "norm_median": pctl(N, .50),
    }


# ----------------------------------------------------------------------------- #
# MONTAGE
# ----------------------------------------------------------------------------- #
def build_montage(records, key, descending, k, out_path, cols=4, cell=224, cap=22):
    if not records:
        return None
    ranked = sorted(records, key=lambda r: r[key], reverse=descending)[:k]
    n = len(ranked)
    rows = (n + cols - 1) // cols
    W, H = cols * cell, rows * (cell + cap)
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    disp_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(cell)])
    for idx, r in enumerate(ranked):
        rr, cc = divmod(idx, cols)
        x0, y0 = cc * cell, rr * (cell + cap)
        try:
            img = disp_tf(Image.open(r["path"]).convert("RGB"))
        except Exception:
            img = Image.new("RGB", (cell, cell), (60, 60, 60))
        canvas.paste(img, (x0, y0))
        draw.rectangle([x0, y0 + cell, x0 + cell, y0 + cell + cap], fill=(0, 0, 0))
        cap_txt = f"L={r['logit']:+.3f}  c={r['cosine']:+.3f}"
        draw.text((x0 + 4, y0 + cell + 4), cap_txt, fill=(240, 240, 240))
    canvas.save(out_path)
    return out_path


# ----------------------------------------------------------------------------- #
# OPTIONAL: LFW auto-fetch (real public faces, one call) — needs scikit-learn
# ----------------------------------------------------------------------------- #
def fetch_lfw_to(out_folder, max_images=800):
    """Dump LFW color face crops to out_folder as PNGs. Run once, then point a bank at it."""
    from sklearn.datasets import fetch_lfw_people
    import numpy as np
    os.makedirs(out_folder, exist_ok=True)
    data = fetch_lfw_people(color=True, resize=1.0, funneled=True,
                            min_faces_per_person=0, slice_=(slice(0, 250), slice(0, 250)))
    imgs = data.images  # [N,H,W,3] floats 0..255
    n = min(max_images, len(imgs))
    for i in range(n):
        arr = imgs[i].astype("uint8")
        Image.fromarray(arr).save(os.path.join(out_folder, f"lfw_{i:05d}.png"))
    print(f"[lfw] wrote {n} face images to {out_folder}")


# ----------------------------------------------------------------------------- #
# MAIN
# ----------------------------------------------------------------------------- #
def main():
    device = get_device()
    print(f"[info] device: {device}   torch {torch.__version__}")
    os.makedirs(OUT_DIR, exist_ok=True)

    model, w, b = load_model(CKPT, device)
    print(f"[head] ||w|| = {w.norm().item():.4f}   bias = {b.item():.4f}")

    tf = make_transform()

    # score each populated bank
    bank_records, bank_stats = {}, {}
    for name, folder in BANKS.items():
        paths = list_images(folder)[:MAX_PER_BANK]
        if not paths:
            print(f"[skip] bank '{name}': no images at '{folder}'")
            continue
        print(f"[scan] bank '{name}': scoring {len(paths)} images ...")
        recs = score_bank(model, w, b, paths, tf, device)
        bank_records[name] = recs
        bank_stats[name] = summarize(recs)

    if not bank_records:
        raise SystemExit(
            "\nNo banks were populated. Edit the BANKS dict to point at folders of "
            "images. Quick start for faces: call fetch_lfw_to('banks/faces') once.")

    # head-σ from reference bank (fallback: pool all banks)
    if REFERENCE_BANK in bank_stats:
        head_sigma = bank_stats[REFERENCE_BANK]["logit_std"]
        sigma_src = REFERENCE_BANK
    else:
        allL = [r["logit"] for recs in bank_records.values() for r in recs]
        head_sigma = st.pstdev(allL) if len(allL) > 1 else 0.0
        sigma_src = "ALL banks pooled"
    print(f"\n[head-σ] = {head_sigma:.4f}  (from '{sigma_src}')")

    # ---- table ----
    print("\n" + "=" * 96)
    print(f"{'bank':<12}{'n':>6}{'logit_med':>11}{'logit_p90':>11}{'logit_max':>11}"
          f"{'cos_med':>10}{'cos_max':>10}{'norm_med':>10}")
    print("-" * 96)
    ranked_banks = sorted(bank_stats.items(),
                          key=lambda kv: kv[1]["logit_median"], reverse=True)
    for name, s in ranked_banks:
        print(f"{name:<12}{s['n']:>6}{s['logit_median']:>11.4f}{s['logit_p90']:>11.4f}"
              f"{s['logit_max']:>11.4f}{s['cos_median']:>10.4f}{s['cos_max']:>10.4f}"
              f"{s['norm_median']:>10.2f}")
    print("=" * 96)

    # ---- separation verdict (logit, primary) ----
    print("\n[separation] banks ranked by MEDIAN LOGIT (high -> low):")
    for i, (name, s) in enumerate(ranked_banks):
        gap = ""
        if i + 1 < len(ranked_banks) and head_sigma > 0:
            nxt = ranked_banks[i + 1][1]["logit_median"]
            gap = f"   gap to next = {(s['logit_median'] - nxt)/head_sigma:+.2f} head-σ"
        print(f"   {i+1}. {name:<12} median={s['logit_median']:+.4f}{gap}")

    if len(ranked_banks) >= 2 and head_sigma > 0:
        top, second = ranked_banks[0], ranked_banks[1]
        sep = (top[1]["logit_median"] - second[1]["logit_median"]) / head_sigma
        print(f"\n[VERDICT] top bank '{top[0]}' leads '{second[0]}' by {sep:.2f} head-σ.")
        if sep > 1.0:
            print("          -> PASSES criterion (>1 head-σ): evidence the positive "
                  f"domain is '{top[0]}'-like. Inspect its top-K montage.")
        else:
            print("          -> does NOT pass (<1 head-σ): no clean domain separation. "
                  "Check cosine ranking + the global bottom-K (polarity may be flipped).")

    # ---- cosine cross-check ----
    ranked_cos = sorted(bank_stats.items(),
                        key=lambda kv: kv[1]["cos_median"], reverse=True)
    print("\n[cosine] banks ranked by MEDIAN COSINE (directional alignment):")
    for i, (name, s) in enumerate(ranked_cos):
        print(f"   {i+1}. {name:<12} cos_median={s['cos_median']:+.4f}")
    if ranked_banks[0][0] == ranked_cos[0][0]:
        print(f"   -> '{ranked_banks[0][0]}' wins on BOTH logit and cosine = genuine "
              "directional alignment (not just high-norm images).")
    else:
        print(f"   -> logit winner ('{ranked_banks[0][0]}') != cosine winner "
              f"('{ranked_cos[0][0]}'): logit lead may be driven by embedding norm. "
              "Trust the cosine winner for DOMAIN identity.")

    # ---- montages ----
    print(f"\n[montage] writing top-{TOPK} and bottom-{TOPK} per bank to {OUT_DIR}/ ...")
    for name, recs in bank_records.items():
        build_montage(recs, "logit", True,  TOPK, f"{OUT_DIR}/{name}_TOP.png")
        build_montage(recs, "logit", False, TOPK, f"{OUT_DIR}/{name}_BOTTOM.png")

    # global montages across all banks (the most informative single artifact)
    all_recs = [r for recs in bank_records.values() for r in recs]
    build_montage(all_recs, "logit", True,  TOPK, f"{OUT_DIR}/_GLOBAL_TOP.png")
    build_montage(all_recs, "logit", False, TOPK, f"{OUT_DIR}/_GLOBAL_BOTTOM.png")
    build_montage(all_recs, "cosine", True, TOPK, f"{OUT_DIR}/_GLOBAL_TOP_COSINE.png")

    # ---- dump stats json ----
    with open(f"{OUT_DIR}/stats.json", "w") as f:
        json.dump({"head_norm": float(w.norm()), "bias": float(b),
                   "head_sigma": head_sigma, "sigma_source": sigma_src,
                   "banks": bank_stats}, f, indent=2)
    print(f"[done] stats -> {OUT_DIR}/stats.json")
    print("\nSEND BACK: the table + [VERDICT] + [cosine] lines above, and the PNGs:")
    print(f"   {OUT_DIR}/_GLOBAL_TOP.png, _GLOBAL_BOTTOM.png, _GLOBAL_TOP_COSINE.png")
    print(f"   plus the TOP/BOTTOM montage for whichever bank wins.")


if __name__ == "__main__":
    # To fetch a quick faces bank first (needs scikit-learn), uncomment:
    # fetch_lfw_to("banks/faces");
    main()