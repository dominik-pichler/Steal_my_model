#!/usr/bin/env python3
"""
Parse the extracted __weights blob as a torchvision ResNet-18 state_dict.
Walks the state_dict in PyTorch's canonical key order, slicing FP32 chunks
from the binary blob.
"""
import sys
import numpy as np
import torch
import torchvision.models as M

BLOB_PATH = "weights.bin"
NUM_CLASSES_CANDIDATES = [1, 2] 

def try_parse(blob: bytes, num_classes: int):
    """Attempt to parse the blob as a ResNet-18 state_dict with the given num_classes."""
    model = M.resnet18(num_classes=num_classes)
    sd = model.state_dict()

    # Only FP32 tensors get serialized; the int64 num_batches_tracked buffers
    # are typically excluded by a hand-rolled dumper. We'll confirm that assumption.
    keys_in_order = [(k, tuple(v.shape), v.numel(), str(v.dtype)) for k, v in sd.items()]
    fp32_keys = [(k, shape, n) for (k, shape, n, dt) in keys_in_order if dt == "torch.float32"]
    int64_keys = [(k, shape, n) for (k, shape, n, dt) in keys_in_order if dt != "torch.float32"]

    expected_fp32 = sum(n for _, _, n in fp32_keys)
    expected_bytes = expected_fp32 * 4

    print(f"\n{'='*70}")
    print(f"Hypothesis: ResNet-18 with num_classes={num_classes}")
    print(f"{'='*70}")
    print(f"  Expected FP32 values: {expected_fp32:,}  ({expected_bytes:,} bytes)")
    print(f"  Actual blob size:     {len(blob)//4:,}  ({len(blob):,} bytes)")
    print(f"  Diff (blob - expected): {len(blob) - expected_bytes:+,} bytes "
          f"({(len(blob) - expected_bytes)//4:+,} floats)")
    print(f"  Excluded int64 buffers in state_dict: {[k for k,_,_ in int64_keys]}")

    # Walk the state_dict, slicing the blob.
    off = 0
    parsed = {}
    issues = []
    print(f"\n  Walking {len(fp32_keys)} FP32 tensors...")
    for k, shape, numel in fp32_keys:
        nbytes = numel * 4
        if off + nbytes > len(blob):
            issues.append(f"OVERFLOW at {k}: need {nbytes} bytes from offset {off}, "
                          f"only {len(blob)-off} remain")
            break
        arr = np.frombuffer(blob[off:off+nbytes], dtype=np.float32).reshape(shape)
        parsed[k] = arr
        off += nbytes

    leftover = len(blob) - off
    print(f"  Parsed to offset {off:,}; leftover at end: {leftover} bytes")

    if issues:
        for i in issues:
            print(f"  ! {i}")
        return parsed, off, leftover, issues

    # Sanity checks on tensors that should have known statistical properties
    print(f"\n  Sanity statistics:")
    for key in ["conv1.weight", "bn1.weight", "bn1.bias",
                "bn1.running_mean", "bn1.running_var",
                "layer4.1.bn2.running_var", "fc.weight", "fc.bias"]:
        if key in parsed:
            a = parsed[key]
            tag = ""
            if "running_var" in key:
                tag = " <- should be strictly positive" if (a >= 0).all() \
                      else " <- !! HAS NEGATIVE VALUES (parse misaligned)"
            elif "weight" in key and "conv" in key:
                tag = " <- conv weights, expect mean~0, std~0.01-0.1"
            print(f"    {key:36s} shape={a.shape}  "
                  f"mean={a.mean():+.4f}  std={a.std():.4f}  "
                  f"min={a.min():+.4f}  max={a.max():+.4f}{tag}")

    return parsed, off, leftover, issues


def main():
    blob = open(BLOB_PATH, "rb").read()
    print(f"Loaded {BLOB_PATH}: {len(blob):,} bytes")
    print(f"First 64 bytes (hex): {blob[:64].hex(' ')}")
    print(f"Last 64 bytes (hex):  {blob[-64:].hex(' ')}")

    if len(blob) % 4 != 0:
        print(f"!! Blob size is not a multiple of 4 ({len(blob)} bytes); "
              "FP32 hypothesis already in trouble")
        return

    results = []
    for nc in NUM_CLASSES_CANDIDATES:
        results.append(try_parse(blob, nc))

    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    for nc, (_, off, leftover, issues) in zip(NUM_CLASSES_CANDIDATES, results):
        verdict = "OK" if not issues and abs(leftover) < 4096 else "MISMATCH"
        print(f"  num_classes={nc}: parsed {off:,} bytes, "
              f"leftover {leftover:+d} bytes -- {verdict}")


if __name__ == "__main__":
    main()
