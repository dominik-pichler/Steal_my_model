import torch

# `obj` is the loaded ResNet from your script; if it's out of scope, reload:
obj = torch.load("reconstructed_model.pth", map_location="cpu", weights_only=False)

w = obj.fc.weight.data.flatten()
b = obj.fc.bias.data

print(f"[head] w shape : {tuple(obj.fc.weight.shape)}")
print(f"[head] b shape : {tuple(obj.fc.bias.shape)}   (should be (1,))")
print(f"[head] ||w||   : {w.norm().item():.4f}")
print(f"[head] bias    : {b.item():.4f}")
print(f"[head] w stats : min={w.min():.4f}  max={w.max():.4f}  mean={w.mean():.4f}")

# How many feature dims carry most of the weight? (sparsity tells us how 'focused' the head is)
wabs = w.abs().sort(descending=True).values
cum = wabs.cumsum(0) / wabs.sum()
k50 = int((cum < 0.5).sum()) + 1
k90 = int((cum < 0.9).sum()) + 1
print(f"[head] {k50} of 512 dims hold 50% of |w|;  {k90} hold 90%")