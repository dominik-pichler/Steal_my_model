#!/usr/bin/env python3
"""
Parse the extracted __weights blob and save it as a proper PyTorch state_dict.
Outputs two files:
  - reconstructed.pth        — the state_dict alone (smaller, framework-agnostic-ish)
  - reconstructed_model.pth  — the full model with weights loaded (loadable in one line)
"""
import numpy as np
import torch
import torchvision.models as M

BLOB_PATH = "weights.bin"
OUT_STATE_DICT = "reconstructed_state_dict.pth"
OUT_FULL_MODEL = "reconstructed_model.pth"

# 1) Read the blob
blob = open(BLOB_PATH, "rb").read()
print(f"Loaded {BLOB_PATH}: {len(blob):,} bytes")

# 2) Build a template ResNet-18 to get the canonical key order and tensor shapes
model = M.resnet18(num_classes=1)
template_sd = model.state_dict()

# 3) Walk the FP32 tensors in order, slicing the blob
fp32_items = [(k, tuple(v.shape), v.numel()) for k, v in template_sd.items()
              if v.dtype == torch.float32]

new_sd = {}
off = 0
for k, shape, numel in fp32_items:
    nbytes = numel * 4
    arr = np.frombuffer(blob[off:off+nbytes], dtype=np.float32).reshape(shape).copy()
    new_sd[k] = torch.from_numpy(arr)
    off += nbytes

print(f"Parsed {len(new_sd)} FP32 tensors, consumed {off:,} bytes "
      f"({len(blob)-off} bytes of trailing zero-padding ignored)")

# 4) Preserve the int64 buffers from the template (num_batches_tracked counters).
#    These weren't in the binary blob, but PyTorch's load_state_dict expects them.
#    I use zeros from the template — they don't affect inference, only training.
for k, v in template_sd.items():
    if v.dtype != torch.float32:
        new_sd[k] = v.clone()

# 5) Validate: try loading into a fresh model
fresh_model = M.resnet18(num_classes=1)
missing, unexpected = fresh_model.load_state_dict(new_sd, strict=True)
print(f"Loaded into a fresh ResNet-18(num_classes=1). "
      f"Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
fresh_model.eval()  # set to inference mode

# 6) Save in two convenient formats
torch.save(new_sd, OUT_STATE_DICT)
torch.save(fresh_model, OUT_FULL_MODEL)
print(f"Saved state_dict to {OUT_STATE_DICT}")
print(f"Saved full model   to {OUT_FULL_MODEL}")

# 7) Quick smoke test: run a zero-image through the model
with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224)
    out = fresh_model(dummy)
print(f"Smoke test: model(zeros) = {out.item():+.6f}  "
      f"(arbitrary number, just confirms the model runs)")
