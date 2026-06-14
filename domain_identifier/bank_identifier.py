import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

CKPT = "reconstructed_model.pth"   # <-- adjust path if needed

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available()
          else "cpu")
print(f"[info] device: {device}")
print(f"[info] torch {torch.__version__}")

# Your own file -> safe to disable the weights_only guard so full modules load.
obj = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"\n[info] top-level type: {type(obj)}")


def find_head(state):
    """Return (weight_key, bias_key) for a [1,512] linear head inside a state_dict."""
    w_key = b_key = None
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and tuple(v.shape) == (1, 512):
            w_key = k
        if isinstance(v, torch.Tensor) and tuple(v.shape) == (1,):
            b_key = k
    return w_key, b_key


backbone = None
head = None

if isinstance(obj, nn.Module):
    # ---- Case A: a full model object was saved ----
    print("\n[info] Saved object is a full nn.Module. Architecture:")
    print(obj)
    model_full = obj.to(device).eval()
    for p in model_full.parameters():
        p.requires_grad_(False)

    # Try to locate the [1,512] linear head for reporting.
    for name, m in model_full.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == 1 and m.in_features == 512:
            print(f"[info] found head linear at: '{name}'  "
                  f"w={tuple(m.weight.shape)} b={tuple(m.weight.shape)}")
    full_model = model_full  # we'll just call this directly

else:
    # ---- Case B: a dict (state_dict or checkpoint wrapper) ----
    if isinstance(obj, dict) and any(k in obj for k in ("state_dict", "model", "model_state_dict")):
        sd = obj.get("state_dict") or obj.get("model_state_dict") or obj.get("model")
        print("[info] Unwrapped a checkpoint dict.")
    else:
        sd = obj
    sd = {k.replace("module.", ""): v for k, v in sd.items()}  # strip DataParallel prefix

    print("\n[info] state_dict keys / shapes:")
    for k, v in sd.items():
        shp = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v)
        print(f"   {k:45s} {shp}")

    w_key, b_key = find_head(sd)
    print(f"\n[info] detected head weight key: {w_key}")
    print(f"[info] detected head bias  key: {b_key}")
    assert w_key is not None, "No [1,512] head weight found — paste the keys above to me."

    # Rebuild a clean, identical backbone + head.
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    # If the checkpoint carries backbone weights, prefer them (guarantees identical g):
    bb_sd = {k: v for k, v in sd.items()
             if k.startswith(("conv1", "bn1", "layer", "fc")) and k not in (w_key, b_key)}
    missing, unexpected = backbone.load_state_dict(
        {k: v for k, v in bb_sd.items() if not k.startswith("fc")}, strict=False)
    print(f"[info] backbone load — missing:{len(missing)} unexpected:{len(unexpected)} "
          f"(0/0 ⇒ stock IMAGENET1K backbone, also fine)")
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    head = nn.Linear(512, 1)
    head.weight.data = sd[w_key].clone().float()
    head.bias.data = (sd[b_key].clone().float() if b_key else torch.zeros(1))
    head = head.to(device).eval()
    for p in head.parameters():
        p.requires_grad_(False)

    def full_model(x):
        return head(backbone(x))

# ---- Sanity check: does f produce a finite scalar logit? ----
x = torch.randn(2, 3, 224, 224, device=device)
with torch.no_grad():
    logit = full_model(x)
print(f"\n[sanity] random-input logit shape: {tuple(logit.shape)}  values: {logit.flatten().tolist()}")
print("[sanity] OK — forward pass works." if logit.numel() == 2 else "[warn] unexpected output shape")

# Report the head geometry — useful for every later rung.
if head is not None:
    w = head.weight.data.flatten()
    print(f"\n[head] ||w|| = {w.norm().item():.4f}   bias = {head.bias.item():.4f}")